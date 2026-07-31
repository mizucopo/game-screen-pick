"""Completed Stage writerのconcurrency test。"""

import hashlib
import json
import threading
from datetime import datetime
from pathlib import Path

import pytest

from src.video_selection.models.completed_stage_bundle import CompletedStageBundle
from src.video_selection.models.processing_stage import ProcessingStage
from src.video_selection.services.build_stage_fingerprint import (
    build_stage_fingerprint,
)
from src.video_selection.services.completed_stage_writer import CompletedStageWriter


def test_same_fingerprint_writes_are_serialized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同じStage Fingerprintの同時書き込みが直列化されること。

    Arrange:
        - first artifact writeを一時停止するfilesystem boundaryが用意される
        - 同じfingerprintへ書く2つのwriter threadが用意される
    Act:
        - first write中にsecond writeが開始される
    Assert:
        - first write解放までsecond writeが完了しないこと
        - 最初に開始したartifactがCompleted Stageとして保持されること
    """
    # Arrange
    cache_folder = tmp_path / "cache"
    subject_fingerprint = "a" * 64
    stage = ProcessingStage.DISCOVER_VIDEO_SET
    semantic_input = {"videos": ["video.mp4"]}
    fingerprint = build_stage_fingerprint(stage, (), semantic_input)
    first_write_started = threading.Event()
    release_first_write = threading.Event()
    second_write_finished = threading.Event()
    errors: list[BaseException] = []
    original_write_bytes = Path.write_bytes

    def controlled_write_bytes(path: Path, content: bytes) -> int:
        if (
            threading.current_thread().name == "first-stage-writer"
            and path.name == "artifact.json"
        ):
            first_write_started.set()
            if not release_first_write.wait(timeout=5):
                msg = "first artifact write was not released"
                raise TimeoutError(msg)
        return original_write_bytes(path, content)

    monkeypatch.setattr(Path, "write_bytes", controlled_write_bytes)

    def write_artifact(value: str, finished: threading.Event | None = None) -> None:
        try:
            CompletedStageWriter(
                cache_folder,
                subject_namespace="video-sets",
                subject_fingerprint=subject_fingerprint,
            ).write(
                stage,
                fingerprint,
                (),
                semantic_input,
                {"value": value},
            )
        except BaseException as error:
            errors.append(error)
        finally:
            if finished is not None:
                finished.set()

    first_thread = threading.Thread(
        target=write_artifact,
        args=("first",),
        name="first-stage-writer",
    )
    second_thread = threading.Thread(
        target=write_artifact,
        args=("second", second_write_finished),
        name="second-stage-writer",
    )

    # Act
    first_thread.start()
    assert first_write_started.wait(timeout=5)
    second_thread.start()
    try:
        second_completed_early = second_write_finished.wait(timeout=0.2)
    finally:
        release_first_write.set()
        first_thread.join(timeout=5)
        second_thread.join(timeout=5)

    # Assert
    assert not second_completed_early
    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert errors == []
    artifact_path = (
        cache_folder
        / "video-sets"
        / subject_fingerprint
        / stage.value
        / fingerprint.value
        / "artifact.json"
    )
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == {"value": "first"}


def test_completed_manifest_describes_content_addressed_artifact(
    tmp_path: Path,
) -> None:
    """Completed Stage manifestに再利用検証情報だけが記録されること。

    Arrange:
        - Video fingerprint、Stage fingerprint、semantic inputが用意される
    Act:
        - videos namespaceへStage artifactが確定される
    Assert:
        - subject、上流、version、相対path、size、hash、完了日時が記録されること
        - absolute cache pathがmanifestへ含まれないこと
    """
    # Arrange
    cache_folder = tmp_path / "private-cache"
    subject_fingerprint = "b" * 64
    stage = ProcessingStage.DISCOVER_VIDEO_SET
    semantic_input = {"decode_backend": "cpu"}
    fingerprint = build_stage_fingerprint(stage, (), semantic_input)
    artifact: dict[str, object] = {"value": "artifact"}
    writer = CompletedStageWriter(
        cache_folder,
        subject_namespace="videos",
        subject_fingerprint=subject_fingerprint,
    )

    # Act
    writer.write(stage, fingerprint, (), semantic_input, artifact)

    # Assert
    stage_folder = (
        cache_folder / "videos" / subject_fingerprint / stage.value / fingerprint.value
    )
    artifact_bytes = (stage_folder / "artifact.json").read_bytes()
    manifest_text = (stage_folder / "manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert set(manifest) == {
        "schema",
        "status",
        "stage",
        "stage_version",
        "stage_fingerprint",
        "subject",
        "upstream_stage_fingerprints",
        "semantic_input",
        "artifacts",
        "completed_at",
    }
    assert manifest["schema"] == "game-screen-pick/completed-stage@1.0.0"
    assert manifest["status"] == "completed"
    assert manifest["stage"] == stage.value
    assert manifest["stage_version"] == "video-set-discovery-v1"
    assert manifest["stage_fingerprint"] == fingerprint.value
    assert manifest["subject"] == {
        "namespace": "videos",
        "fingerprint": subject_fingerprint,
    }
    assert manifest["upstream_stage_fingerprints"] == []
    assert manifest["semantic_input"] == semantic_input
    assert manifest["artifacts"] == [
        {
            "path": "artifact.json",
            "size_bytes": len(artifact_bytes),
            "sha256": hashlib.sha256(artifact_bytes).hexdigest(),
        }
    ]
    assert datetime.fromisoformat(manifest["completed_at"]).tzinfo is not None
    assert str(cache_folder) not in manifest_text


@pytest.mark.parametrize(
    "checkpoint",
    ["before-manifest", "after-manifest", "before-rename"],
)
def test_fault_before_atomic_commit_leaves_no_completed_stage(
    tmp_path: Path,
    checkpoint: str,
) -> None:
    """atomic commit前のfaultでCompleted Stageが観測されないこと。

    Arrange:
        - manifestとrenameのcommit pointへfault injectorが設定される
    Act:
        - Stage artifactの確定が試行される
    Assert:
        - faultが返されfinal entryとtemporary entryが残らないこと
    """
    # Arrange
    cache_folder = tmp_path / "cache"
    subject_fingerprint = "c" * 64
    stage = ProcessingStage.DISCOVER_VIDEO_SET
    semantic_input = {"value": checkpoint}
    fingerprint = build_stage_fingerprint(stage, (), semantic_input)

    def inject_fault(actual_checkpoint: str) -> None:
        if actual_checkpoint == checkpoint:
            raise OSError(f"injected {checkpoint}")

    writer = CompletedStageWriter(
        cache_folder,
        subject_namespace="video-sets",
        subject_fingerprint=subject_fingerprint,
        fault_injector=inject_fault,
    )

    # Act
    # Assert
    with pytest.raises(OSError, match=f"injected {checkpoint}"):
        writer.write(stage, fingerprint, (), semantic_input, {"value": "fresh"})
    stage_root = cache_folder / "video-sets" / subject_fingerprint / stage.value
    assert not (stage_root / fingerprint.value).exists()
    assert tuple(stage_root.glob("*.tmp")) == ()


def test_fault_after_rename_keeps_reusable_completed_stage(tmp_path: Path) -> None:
    """rename後のfaultでもatomicに確定済みのStageが再利用されること。

    Arrange:
        - after-rename checkpointだけで失敗するwriterが用意される
    Act:
        - Stage writeがrename後に失敗し通常writerからreadされる
    Assert:
        - 完全なartifactがCompleted Stageとして再利用されること
    """
    # Arrange
    cache_folder = tmp_path / "cache"
    subject_fingerprint = "d" * 64
    stage = ProcessingStage.DISCOVER_VIDEO_SET
    semantic_input = {"value": "after-rename"}
    fingerprint = build_stage_fingerprint(stage, (), semantic_input)

    def inject_fault(checkpoint: str) -> None:
        if checkpoint == "after-rename":
            raise OSError("injected after-rename")

    failing_writer = CompletedStageWriter(
        cache_folder,
        subject_namespace="video-sets",
        subject_fingerprint=subject_fingerprint,
        fault_injector=inject_fault,
    )

    # Act
    with pytest.raises(OSError, match="injected after-rename"):
        failing_writer.write(
            stage,
            fingerprint,
            (),
            semantic_input,
            {"value": "committed"},
        )
    restored = CompletedStageWriter(
        cache_folder,
        subject_namespace="video-sets",
        subject_fingerprint=subject_fingerprint,
    ).read(stage, fingerprint, (), semantic_input)

    # Assert
    assert restored == {"value": "committed"}


def test_artifact_permission_failure_preserves_completed_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """一時的な読込権限失敗でCompleted Stageが削除されないこと。

    Arrange:
        - 完全なCompleted Stageとartifactだけを拒否する障害が用意される
    Act:
        - 同じStageの再確定が試行される
    Assert:
        - producerは実行されず既存artifact bytesが保持されること
    """
    # Arrange
    cache_folder = tmp_path / "cache"
    subject_fingerprint = "4" * 64
    stage = ProcessingStage.SCAN_VIDEO
    semantic_input = {"value": "stable"}
    fingerprint = build_stage_fingerprint(stage, (), semantic_input)
    writer = CompletedStageWriter(
        cache_folder,
        subject_namespace="videos",
        subject_fingerprint=subject_fingerprint,
    )
    writer.write(stage, fingerprint, (), semantic_input, {"value": "stable"})
    artifact_path = (
        cache_folder
        / "videos"
        / subject_fingerprint
        / stage.value
        / fingerprint.value
        / "artifact.json"
    )
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
        writer.write_artifacts(
            stage,
            fingerprint,
            (),
            semantic_input,
            produce,
        )
    assert producer_calls == 0
    assert original_read_bytes(artifact_path) == b'{\n  "value": "stable"\n}\n'


def test_validator_permission_failure_preserves_completed_stage(
    tmp_path: Path,
) -> None:
    """validatorのaccess障害でCompleted Stageが削除されないこと。

    Arrange:
        - 完全なCompleted StageとPermissionErrorになるvalidatorが用意される
    Act:
        - 同じfingerprintのStage確定が再試行される
    Assert:
        - access障害が返され、producerとartifact bytesが変更されないこと
    """
    # Arrange
    cache_folder = tmp_path / "cache"
    subject_fingerprint = "1" * 64
    stage = ProcessingStage.EXTRACT_FRAME_CANDIDATES
    semantic_input = {"algorithm": "stable"}
    fingerprint = build_stage_fingerprint(stage, (), semantic_input)
    writer = CompletedStageWriter(
        cache_folder,
        subject_namespace="videos",
        subject_fingerprint=subject_fingerprint,
    )
    producer_calls = 0

    def produce(_stage_folder: Path) -> dict[str, object]:
        nonlocal producer_calls
        producer_calls += 1
        return {"schema": "valid"}

    writer.write_artifacts(stage, fingerprint, (), semantic_input, produce)
    artifact_path = (
        cache_folder
        / "videos"
        / subject_fingerprint
        / stage.value
        / fingerprint.value
        / "artifact.json"
    )
    original_bytes = artifact_path.read_bytes()

    def deny_validation(_bundle: CompletedStageBundle) -> None:
        raise PermissionError("injected validator permission failure")

    # Act
    # Assert
    with pytest.raises(
        PermissionError,
        match="injected validator permission failure",
    ):
        writer.write_artifacts(
            stage,
            fingerprint,
            (),
            semantic_input,
            produce,
            validate_bundle=deny_validation,
        )
    assert producer_calls == 1
    assert artifact_path.read_bytes() == original_bytes


@pytest.mark.parametrize("stage", tuple(ProcessingStage))
@pytest.mark.parametrize(
    ("checkpoint", "expected_reusable"),
    [("before-rename", False), ("after-rename", True)],
)
def test_each_processing_stage_observes_atomic_commit_boundary(
    tmp_path: Path,
    stage: ProcessingStage,
    checkpoint: str,
    expected_reusable: bool,
) -> None:
    """全Processing Stageがrename境界の前後で同じ再利用契約を守ること。

    Arrange:
        - 指定Stageのatomic rename直前または直後で失敗するwriterが用意される
    Act:
        - Stage確定が失敗した後に通常writerから同じfingerprintが読まれる
    Assert:
        - rename前は未完了、rename後だけCompleted Stageとして再利用されること
    """
    # Arrange
    cache_folder = tmp_path / "cache"
    subject_fingerprint = "5" * 64
    semantic_input = {"stage": stage.value}
    fingerprint = build_stage_fingerprint(stage, (), semantic_input)

    def inject_fault(actual_checkpoint: str) -> None:
        if actual_checkpoint == checkpoint:
            raise OSError(f"injected {checkpoint}")

    writer = CompletedStageWriter(
        cache_folder,
        subject_namespace="video-sets",
        subject_fingerprint=subject_fingerprint,
        fault_injector=inject_fault,
    )

    # Act
    with pytest.raises(OSError, match=f"injected {checkpoint}"):
        writer.write(
            stage,
            fingerprint,
            (),
            semantic_input,
            {"value": "committed"},
        )
    restored = CompletedStageWriter(
        cache_folder,
        subject_namespace="video-sets",
        subject_fingerprint=subject_fingerprint,
    ).read(stage, fingerprint, (), semantic_input)

    # Assert
    assert (restored is not None) is expected_reusable


def test_recognized_orphan_temporary_stage_is_removed_before_recompute(
    tmp_path: Path,
) -> None:
    """hard terminationで残った認識可能なpartial Stageだけが削除されること。

    Arrange:
        - Stage rootに正規形式のorphan temporary folderと未知entryが用意される
    Act:
        - 同じStage Fingerprintが再計算されCompleted Stageへ確定される
    Assert:
        - orphanだけが削除され、未知entryとCompleted Stageが保持されること
    """
    # Arrange
    cache_folder = tmp_path / "cache"
    subject_fingerprint = "1" * 64
    stage = ProcessingStage.DISCOVER_VIDEO_SET
    semantic_input = {"value": "resume"}
    fingerprint = build_stage_fingerprint(stage, (), semantic_input)
    stage_root = cache_folder / "video-sets" / subject_fingerprint / stage.value
    stage_root.mkdir(parents=True)
    orphan = stage_root / f".{fingerprint.value}.{'a' * 32}.tmp"
    orphan.mkdir()
    (orphan / "partial.bin").write_bytes(b"partial")
    unknown = stage_root / f".{fingerprint.value}.user.tmp"
    unknown.mkdir()
    (unknown / "keep.bin").write_bytes(b"keep")
    writer = CompletedStageWriter(
        cache_folder,
        subject_namespace="video-sets",
        subject_fingerprint=subject_fingerprint,
    )

    # Act
    writer.write(stage, fingerprint, (), semantic_input, {"value": "recomputed"})

    # Assert
    assert (
        orphan.exists(),
        unknown.exists(),
        (stage_root / fingerprint.value / "manifest.json").is_file(),
    ) == (False, True, True)


def test_rename_failure_leaves_no_partial_completed_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """filesystem rename失敗時にpartial Completed Stageが残らないこと。

    Arrange:
        - temporary Stage directoryのrenameがdisk errorになる
    Act:
        - Stage writeが実行される
    Assert:
        - errorが返されfinal entryとtemporary entryが残らないこと
    """
    # Arrange
    cache_folder = tmp_path / "cache"
    subject_fingerprint = "e" * 64
    stage = ProcessingStage.DISCOVER_VIDEO_SET
    semantic_input = {"value": "rename"}
    fingerprint = build_stage_fingerprint(stage, (), semantic_input)
    original_replace = Path.replace

    def fail_temporary_replace(path: Path, target: Path) -> Path:
        if path.name.endswith(".tmp"):
            raise OSError("injected rename failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_temporary_replace)
    writer = CompletedStageWriter(
        cache_folder,
        subject_namespace="video-sets",
        subject_fingerprint=subject_fingerprint,
    )

    # Act
    # Assert
    with pytest.raises(OSError, match="injected rename failure"):
        writer.write(stage, fingerprint, (), semantic_input, {"value": "fresh"})
    stage_root = cache_folder / "video-sets" / subject_fingerprint / stage.value
    assert not (stage_root / fingerprint.value).exists()
    assert tuple(stage_root.glob("*.tmp")) == ()


@pytest.mark.parametrize(
    ("target_name", "failure"),
    [
        pytest.param("artifact.json", OSError("disk full"), id="disk-full"),
        pytest.param(
            "manifest.json",
            PermissionError("permission denied"),
            id="permission-denied",
        ),
    ],
)
def test_artifact_or_manifest_write_failure_is_not_reusable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_name: str,
    failure: OSError,
) -> None:
    """artifactまたはmanifest書込失敗が再利用可能entryを残さないこと。

    Arrange:
        - artifactにdisk failureまたはmanifestにpermission failureが注入される
    Act:
        - Stage writeが実行される
    Assert:
        - filesystem errorが返されCompleted Stageが存在しないこと
    """
    # Arrange
    cache_folder = tmp_path / "cache"
    subject_fingerprint = "f" * 64
    stage = ProcessingStage.DISCOVER_VIDEO_SET
    semantic_input = {"target": target_name}
    fingerprint = build_stage_fingerprint(stage, (), semantic_input)
    original_write_bytes = Path.write_bytes

    def fail_target_write(path: Path, data: bytes) -> int:
        if path.name == target_name:
            raise failure
        return original_write_bytes(path, data)

    monkeypatch.setattr(Path, "write_bytes", fail_target_write)
    writer = CompletedStageWriter(
        cache_folder,
        subject_namespace="videos",
        subject_fingerprint=subject_fingerprint,
    )

    # Act
    # Assert
    with pytest.raises(type(failure), match=str(failure)):
        writer.write(stage, fingerprint, (), semantic_input, {"value": "fresh"})
    stage_root = cache_folder / "videos" / subject_fingerprint / stage.value
    assert not (stage_root / fingerprint.value).exists()
    assert tuple(stage_root.glob("*.tmp")) == ()


def test_multi_artifact_stage_requires_every_artifact_to_be_intact(
    tmp_path: Path,
) -> None:
    """複数artifactがすべて健全な場合だけCompleted Stageが復元されること。

    Arrange:
        - JSON artifactと2件のproxy画像を生成するproducerが用意される
    Act:
        - Stageが確定され、片方のproxyが後から破損される
    Assert:
        - 確定直後はartifactとStage rootが復元されること
        - 1件でも破損した後はStage全体が再利用されないこと
    """
    # Arrange
    cache_folder = tmp_path / "cache"
    subject_fingerprint = "1" * 64
    stage = ProcessingStage.EXTRACT_FRAME_CANDIDATES
    semantic_input = {"algorithm": "candidate-v1"}
    fingerprint = build_stage_fingerprint(stage, (), semantic_input)
    writer = CompletedStageWriter(
        cache_folder,
        subject_namespace="videos",
        subject_fingerprint=subject_fingerprint,
    )

    def produce_artifacts(stage_folder: Path) -> dict[str, object]:
        proxy_folder = stage_folder / "candidates"
        proxy_folder.mkdir()
        (proxy_folder / "first.jpg").write_bytes(b"first-proxy")
        (proxy_folder / "second.jpg").write_bytes(b"second-proxy")
        return {
            "candidate_proxy_paths": [
                "candidates/first.jpg",
                "candidates/second.jpg",
            ]
        }

    # Act
    writer.write_artifacts(
        stage,
        fingerprint,
        (),
        semantic_input,
        produce_artifacts,
    )
    restored = writer.read_bundle(stage, fingerprint, (), semantic_input)

    # Assert
    assert restored is not None
    assert restored.artifact == {
        "candidate_proxy_paths": [
            "candidates/first.jpg",
            "candidates/second.jpg",
        ]
    }
    assert restored.root.joinpath("candidates/first.jpg").read_bytes() == b"first-proxy"

    # Act
    restored.root.joinpath("candidates/second.jpg").write_bytes(b"corrupt")

    # Assert
    assert writer.read_bundle(stage, fingerprint, (), semantic_input) is None


def test_domain_invalid_completed_stage_is_rebuilt_under_same_fingerprint(
    tmp_path: Path,
) -> None:
    """hash整合性があってもdomain不正なCompleted Stageが再構築されること。

    Arrange:
        - validator付きで一つのCompleted Stageが確定される
        - artifact schemaと対応manifest hashが不正schemaへ更新される
    Act:
        - 同じfingerprintがvalidator付きで再度確定される
    Assert:
        - 不正Stageだけが削除されproducerから再構築されること
    """
    # Arrange
    cache_folder = tmp_path / "cache"
    subject_fingerprint = "6" * 64
    stage = ProcessingStage.EXTRACT_FRAME_CANDIDATES
    semantic_input = {"algorithm": "candidate-v1"}
    fingerprint = build_stage_fingerprint(stage, (), semantic_input)
    writer = CompletedStageWriter(
        cache_folder,
        subject_namespace="videos",
        subject_fingerprint=subject_fingerprint,
    )
    calls = 0

    def produce(_stage_folder: Path) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"schema": "valid", "call": calls}

    def validate(bundle: CompletedStageBundle) -> None:
        if bundle.artifact.get("schema") != "valid":
            raise ValueError("domain invalid")

    writer.write_artifacts(
        stage,
        fingerprint,
        (),
        semantic_input,
        produce,
        validate_bundle=validate,
    )
    stage_folder = (
        cache_folder / "videos" / subject_fingerprint / stage.value / fingerprint.value
    )
    invalid_bytes = b'{"call":1,"schema":"invalid"}'
    (stage_folder / "artifact.json").write_bytes(invalid_bytes)
    manifest_path = stage_folder / "manifest.json"
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
    writer.write_artifacts(
        stage,
        fingerprint,
        (),
        semantic_input,
        produce,
        validate_bundle=validate,
    )
    restored = writer.read_bundle(stage, fingerprint, (), semantic_input)

    # Assert
    assert calls == 2
    assert restored is not None
    assert restored.artifact == {"schema": "valid", "call": 2}
