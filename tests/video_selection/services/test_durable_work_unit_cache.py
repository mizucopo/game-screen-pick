"""Durable Work Unit cacheの契約test。"""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from src.video_selection.models.checkpoint_operation import CheckpointOperation
from src.video_selection.models.durable_work_unit_bundle import (
    DurableWorkUnitBundle,
)
from src.video_selection.services import durable_work_unit_cache
from src.video_selection.services.durable_work_unit_cache import (
    DurableWorkUnitCache,
)


def test_completed_work_unit_is_reused_without_calling_producer(
    tmp_path: Path,
) -> None:
    """完了済みWork Unitがproducerを再実行せず復元されること。

    Arrange:
        - 一つのWork Unitを確定するcacheとproducerが用意される
    Act:
        - 同じ意味入力でWork Unitが2回resolveされる
    Assert:
        - producerは1回だけ実行され、2回目が再利用として返されること
    """
    # Arrange
    calls = 0
    cache = DurableWorkUnitCache(
        tmp_path / "cache",
        subject_fingerprint="a" * 64,
        operation=CheckpointOperation.FRAME_REFINEMENT_GROUP,
    )

    def produce(folder: Path) -> dict[str, object]:
        nonlocal calls
        calls += 1
        (folder / "payload.bin").write_bytes(b"payload")
        return {"value": "stable"}

    # Act
    first, first_reused = cache.resolve("unit-1", {"option": 1}, produce)
    second, second_reused = cache.resolve("unit-1", {"option": 1}, produce)

    # Assert
    assert calls == 1
    assert (first_reused, second_reused) == (False, True)
    assert first.artifact == second.artifact == {"value": "stable"}
    assert second.root.joinpath("payload.bin").read_bytes() == b"payload"


@pytest.mark.parametrize(
    ("checkpoint", "expected_reused"),
    [
        ("before-manifest", False),
        ("after-manifest", False),
        ("before-rename", False),
        ("after-rename", True),
    ],
)
def test_fault_boundary_preserves_only_atomically_committed_work(
    tmp_path: Path,
    checkpoint: str,
    expected_reused: bool,
) -> None:
    """atomic rename後に確定したWork Unitだけが再開時に再利用されること。

    Arrange:
        - 指定commit境界で失敗するWork Unit cacheが用意される
    Act:
        - 初回resolve失敗後に通常cacheで同じWork Unitがresolveされる
    Assert:
        - rename後だけproducerが再実行されず、それ以前は再計算されること
    """
    # Arrange
    calls = 0

    def inject_fault(actual: str) -> None:
        if actual == checkpoint:
            raise OSError(f"injected {checkpoint}")

    def produce(_folder: Path) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"call": calls}

    failing = DurableWorkUnitCache(
        tmp_path / "cache",
        subject_fingerprint="b" * 64,
        operation=CheckpointOperation.FRAME_REFINEMENT_GROUP,
        fault_injector=inject_fault,
    )

    # Act
    with pytest.raises(OSError, match=f"injected {checkpoint}"):
        failing.resolve("unit", {"stable": True}, produce)
    recovered, reused = DurableWorkUnitCache(
        tmp_path / "cache",
        subject_fingerprint="b" * 64,
        operation=CheckpointOperation.FRAME_REFINEMENT_GROUP,
    ).resolve("unit", {"stable": True}, produce)

    # Assert
    assert reused is expected_reused
    assert calls == (1 if expected_reused else 2)
    assert recovered.artifact == {"call": 1 if expected_reused else 2}


def test_corrupt_artifact_recomputes_only_target_work_unit(tmp_path: Path) -> None:
    """破損したWork Unitだけが再計算され、健全な兄弟は再利用されること。

    Arrange:
        - 二つのWork Unitが確定され、片方のartifactだけが破損される
    Act:
        - 両Work Unitが再度resolveされる
    Assert:
        - 破損した単位だけproducerが再実行されること
    """
    # Arrange
    calls: list[str] = []
    cache = DurableWorkUnitCache(
        tmp_path / "cache",
        subject_fingerprint="c" * 64,
        operation=CheckpointOperation.FRAME_REFINEMENT_GROUP,
    )

    def produce(key: str) -> Callable[[Path], dict[str, object]]:
        def write(_folder: Path) -> dict[str, object]:
            calls.append(key)
            return {"key": key}

        return write

    first, _ = cache.resolve("first", {}, produce("first"))
    cache.resolve("second", {}, produce("second"))
    first.root.joinpath("artifact.json").write_text("corrupt", encoding="utf-8")

    # Act
    _, first_reused = cache.resolve("first", {}, produce("first"))
    _, second_reused = cache.resolve("second", {}, produce("second"))

    # Assert
    assert calls == ["first", "second", "first"]
    assert (first_reused, second_reused) == (False, True)


def test_artifact_permission_failure_preserves_completed_work_unit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """一時的な読込権限失敗でCompleted Work Unitが削除されないこと。

    Arrange:
        - 完全なWork Unitとartifactだけを拒否するfilesystem障害が用意される
    Act:
        - 同じWork Unitが再度resolveされる
    Assert:
        - producerは再実行されず既存checkpoint bytesが保持されること
    """
    # Arrange
    cache = DurableWorkUnitCache(
        tmp_path / "cache",
        subject_fingerprint="9" * 64,
        operation=CheckpointOperation.FRAME_REFINEMENT_GROUP,
    )
    completed, _ = cache.resolve("unit", {}, lambda _folder: {"value": "stable"})
    artifact_path = completed.root / "artifact.json"
    original_read_bytes = Path.read_bytes
    producer_calls = 0

    def deny_artifact_read(path: Path) -> bytes:
        if path == artifact_path:
            raise PermissionError("injected permission failure")
        return original_read_bytes(path)

    def produce(_folder: Path) -> dict[str, object]:
        nonlocal producer_calls
        producer_calls += 1
        return {"value": "replacement"}

    monkeypatch.setattr(Path, "read_bytes", deny_artifact_read)

    # Act
    # Assert
    with pytest.raises(PermissionError, match="injected permission failure"):
        cache.resolve("unit", {}, produce)
    assert producer_calls == 0
    assert original_read_bytes(artifact_path) == b'{\n  "value": "stable"\n}\n'


def test_validator_permission_failure_preserves_completed_work_unit(
    tmp_path: Path,
) -> None:
    """validatorのaccess障害でCompleted Work Unitが削除されないこと。

    Arrange:
        - domain検証済みのWork UnitとPermissionErrorになるvalidatorが用意される
    Act:
        - 同じWork Unitが再度resolveされる
    Assert:
        - access障害が返され、producerとcheckpoint bytesが変更されないこと
    """
    # Arrange
    cache = DurableWorkUnitCache(
        tmp_path / "cache",
        subject_fingerprint="e" * 64,
        operation=CheckpointOperation.FRAME_REFINEMENT_GROUP,
    )
    producer_calls = 0

    def produce(_folder: Path) -> dict[str, object]:
        nonlocal producer_calls
        producer_calls += 1
        return {"schema": "valid"}

    completed, _ = cache.resolve("unit", {}, produce)
    artifact_path = completed.root / "artifact.json"
    original_bytes = artifact_path.read_bytes()

    def deny_validation(_bundle: DurableWorkUnitBundle) -> None:
        raise PermissionError("injected validator permission failure")

    # Act
    # Assert
    with pytest.raises(
        PermissionError,
        match="injected validator permission failure",
    ):
        cache.resolve(
            "unit",
            {},
            produce,
            validate_bundle=deny_validation,
        )
    assert producer_calls == 1
    assert artifact_path.read_bytes() == original_bytes


def test_domain_invalid_artifact_recomputes_only_target_work_unit(
    tmp_path: Path,
) -> None:
    """hash整合性があってもdomain不正なWork Unitだけが再計算されること。

    Arrange:
        - 二つのWork Unitが確定され、片方だけ不正schemaへ改変される
        - 改変後byteに合わせてmanifest hashも更新される
    Act:
        - domain validator付きで両Work Unitが再度resolveされる
    Assert:
        - domain不正な単位だけproducerが再実行されること
    """
    # Arrange
    calls: list[str] = []
    cache = DurableWorkUnitCache(
        tmp_path / "cache",
        subject_fingerprint="f" * 64,
        operation=CheckpointOperation.FRAME_REFINEMENT_GROUP,
    )

    def produce(key: str) -> Callable[[Path], dict[str, object]]:
        def write(_folder: Path) -> dict[str, object]:
            calls.append(key)
            return {"schema": "valid", "key": key}

        return write

    def validate(bundle: DurableWorkUnitBundle) -> None:
        if bundle.artifact.get("schema") != "valid":
            raise ValueError("domain invalid")

    first, _ = cache.resolve(
        "first",
        {},
        produce("first"),
        validate_bundle=validate,
    )
    cache.resolve(
        "second",
        {},
        produce("second"),
        validate_bundle=validate,
    )
    artifact_path = first.root / "artifact.json"
    invalid_bytes = b'{"key":"first","schema":"invalid"}'
    artifact_path.write_bytes(invalid_bytes)
    manifest_path = first.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_record = next(
        item for item in manifest["artifacts"] if item["path"] == "artifact.json"
    )
    artifact_record["size_bytes"] = len(invalid_bytes)
    artifact_record["sha256"] = hashlib.sha256(invalid_bytes).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Act
    _, first_reused = cache.resolve(
        "first",
        {},
        produce("first"),
        validate_bundle=validate,
    )
    _, second_reused = cache.resolve(
        "second",
        {},
        produce("second"),
        validate_bundle=validate,
    )

    # Assert
    assert calls == ["first", "second", "first"]
    assert (first_reused, second_reused) == (False, True)


def test_engine_version_change_invalidates_work_unit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """処理engine version変更時に同じWork Unitが再計算されること。

    Arrange:
        - version 1で確定済みのWork Unitが用意される
    Act:
        - version 2のcacheで同じkeyと意味入力がresolveされる
    Assert:
        - 古いcheckpointは再利用されず新versionで確定されること
    """
    # Arrange
    calls = 0

    def produce(_folder: Path) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"call": calls}

    DurableWorkUnitCache(
        tmp_path / "cache",
        subject_fingerprint="d" * 64,
        operation=CheckpointOperation.FRAME_REFINEMENT_GROUP,
    ).resolve("unit", {}, produce)
    monkeypatch.setattr(
        durable_work_unit_cache,
        "checkpoint_version",
        lambda _operation: "frame-refinement-group-v2",
    )

    # Act
    result, reused = DurableWorkUnitCache(
        tmp_path / "cache",
        subject_fingerprint="d" * 64,
        operation=CheckpointOperation.FRAME_REFINEMENT_GROUP,
    ).resolve("unit", {}, produce)

    # Assert
    assert reused is False
    assert calls == 2
    assert result.artifact == {"call": 2}


def test_manifest_contains_no_absolute_cache_path(tmp_path: Path) -> None:
    """Work Unit manifestにabsolute cache pathが保存されないこと。

    Arrange:
        - private cache配下へ一つのWork Unitが確定される
    Act:
        - manifest JSONが読み取られる
    Assert:
        - absolute pathを含まず再利用に必要な情報だけが記録されること
    """
    # Arrange
    cache_root = tmp_path / "private-cache"
    cache = DurableWorkUnitCache(
        cache_root,
        subject_fingerprint="e" * 64,
        operation=CheckpointOperation.FRAME_REFINEMENT_GROUP,
    )

    # Act
    bundle, _ = cache.resolve(
        "unit-1",
        {"range": [0, 10]},
        lambda _folder: {"value": "ok"},
    )
    manifest_text = bundle.root.joinpath("manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)

    # Assert
    assert str(cache_root) not in manifest_text
    assert manifest["work_unit_key"] == "unit-1"
    assert manifest["semantic_input"] == {"range": [0, 10]}
