"""Processing cache preparationのreal filesystem test。"""

import json
import shutil
from pathlib import Path

import pytest

from src.video_selection.services.input_folder_lock import InputFolderLock
from src.video_selection.services.prepare_processing_cache import (
    prepare_processing_cache,
)
from src.video_selection.vision.vision_contract import (
    CANDIDATE_ANNOTATION_STAGE_CONTRACT_VERSION,
)


def test_recognized_legacy_cache_is_deleted_and_new_cache_is_preserved(
    tmp_path: Path,
) -> None:
    """認識済みlegacyだけが削除され新cacheとunknown entryが保持されること。

    Arrange:
        - legacy三種、新cache三種、unknown entryがcache rootに用意される
    Act:
        - Input Lock内でprocessing cacheが準備される
    Assert:
        - legacyだけが削除され件数とbyteがstructuredに返されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    cache_folder = input_folder / ".game-screen-pick" / "cache"
    neutral_cache = cache_folder / "neutral-analysis"
    neutral_cache.mkdir(parents=True)
    (neutral_cache / "one.json").write_bytes(b"1234")
    legacy_scene = cache_folder / "ollama-scenes.json"
    legacy_scene.write_bytes(b"123456")
    legacy_identities = cache_folder / "video-identities"
    legacy_identities.mkdir()
    (legacy_identities / "entry.json").write_bytes(b"12345")
    (cache_folder / "videos" / "fingerprint").mkdir(parents=True)
    (cache_folder / "video-sets" / "fingerprint").mkdir(parents=True)
    (cache_folder / "work-units" / "fingerprint").mkdir(parents=True)
    (cache_folder / "unknown" / "keep").mkdir(parents=True)

    # Act
    with InputFolderLock(input_folder) as input_lock:
        diagnostic = prepare_processing_cache(
            cache_folder,
            input_lock=input_lock,
            reset_cache=False,
        )

    # Assert
    assert diagnostic.removed_entry_count == 3
    assert diagnostic.removed_bytes == 15
    assert not neutral_cache.exists()
    assert not legacy_scene.exists()
    assert not legacy_identities.exists()
    assert (cache_folder / "videos" / "fingerprint").is_dir()
    assert (cache_folder / "video-sets" / "fingerprint").is_dir()
    assert (cache_folder / "work-units" / "fingerprint").is_dir()
    assert (cache_folder / "unknown" / "keep").is_dir()


def test_legacy_candidate_annotation_contract_cache_is_deleted(
    tmp_path: Path,
) -> None:
    """旧Stage ContractのCandidate Annotationだけが削除されること。

    Arrange:
        - 同じVideo Setに旧versionと現行versionのAnnotation cacheが用意される
        - 同じStage rootに認識できないentryが用意される
    Act:
        - Input Lock内でprocessing cacheが準備される
    Assert:
        - 旧versionだけが削除され、現行versionとunknown entryが保持されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    cache_folder = input_folder / ".game-screen-pick" / "cache"
    subject_fingerprint = "a" * 64
    stage_root = (
        cache_folder / "video-sets" / subject_fingerprint / "annotate-candidate"
    )
    legacy_fingerprint = "b" * 64
    current_fingerprint = "c" * 64
    legacy_folder = stage_root / legacy_fingerprint
    current_folder = stage_root / current_fingerprint
    unknown_folder = stage_root / "unknown-entry"
    for folder, fingerprint, version in (
        (
            legacy_folder,
            legacy_fingerprint,
            "candidate-annotation-stage-v31",
        ),
        (
            current_folder,
            current_fingerprint,
            CANDIDATE_ANNOTATION_STAGE_CONTRACT_VERSION,
        ),
    ):
        folder.mkdir(parents=True)
        (folder / "artifact.json").write_text("{}", encoding="utf-8")
        (folder / "manifest.json").write_text(
            json.dumps(
                {
                    "schema": "game-screen-pick/completed-stage@1.0.0",
                    "status": "completed",
                    "stage": "annotate-candidate",
                    "stage_fingerprint": fingerprint,
                    "subject": {
                        "namespace": "video-sets",
                        "fingerprint": subject_fingerprint,
                    },
                    "semantic_input": {
                        "stage_contract_version": version,
                    },
                }
            ),
            encoding="utf-8",
        )
    unknown_folder.mkdir()
    (unknown_folder / "keep").write_text("unknown", encoding="utf-8")
    legacy_bytes = sum(
        path.stat().st_size for path in legacy_folder.iterdir() if path.is_file()
    )

    # Act
    with InputFolderLock(input_folder) as input_lock:
        diagnostic = prepare_processing_cache(
            cache_folder,
            input_lock=input_lock,
            reset_cache=False,
        )

    # Assert
    assert diagnostic.removed_entry_count == 1
    assert diagnostic.removed_bytes == legacy_bytes
    assert not legacy_folder.exists()
    assert current_folder.is_dir()
    assert unknown_folder.is_dir()


def test_legacy_cleanup_failure_is_fatal_and_new_cache_is_untouched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """legacy削除失敗がfatalとなり新cacheへ変更が加えられないこと。

    Arrange:
        - legacy directoryと新しいvideos cacheが用意される
        - legacy directory削除だけがpermission errorになる
    Act:
        - processing cache preparationが実行される
    Assert:
        - errorが返され新cache artifactが保持されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    cache_folder = input_folder / ".game-screen-pick" / "cache"
    legacy_cache = cache_folder / "neutral-analysis"
    legacy_cache.mkdir(parents=True)
    new_artifact = cache_folder / "videos" / "fingerprint" / "artifact.json"
    new_artifact.parent.mkdir(parents=True)
    new_artifact.write_text("keep", encoding="utf-8")
    original_rmtree = shutil.rmtree

    def fail_legacy_delete(path: Path) -> None:
        if Path(path) == legacy_cache:
            raise PermissionError("injected legacy permission failure")
        original_rmtree(path)

    monkeypatch.setattr(shutil, "rmtree", fail_legacy_delete)

    # Act
    # Assert
    with (
        InputFolderLock(input_folder) as input_lock,
        pytest.raises(PermissionError, match="injected legacy"),
    ):
        prepare_processing_cache(
            cache_folder,
            input_lock=input_lock,
            reset_cache=False,
        )
    assert new_artifact.read_text(encoding="utf-8") == "keep"


def test_write_preflight_failure_does_not_delete_legacy_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cache書込preflight失敗時にlegacy削除が開始されないこと。

    Arrange:
        - legacy cacheとwrite probeのpermission failureが用意される
    Act:
        - Input Lock内でprocessing cache preparationが実行される
    Assert:
        - legacy cacheが削除されずfatal errorになること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    cache_folder = input_folder / ".game-screen-pick" / "cache"
    legacy_cache = cache_folder / "neutral-analysis"
    legacy_cache.mkdir(parents=True)
    legacy_artifact = legacy_cache / "keep.json"
    legacy_artifact.write_text("keep", encoding="utf-8")
    original_write_bytes = Path.write_bytes

    def fail_write_probe(path: Path, data: bytes) -> int:
        if path.name.startswith(".write-probe-"):
            raise PermissionError("injected cache write failure")
        return original_write_bytes(path, data)

    monkeypatch.setattr(Path, "write_bytes", fail_write_probe)

    # Act
    # Assert
    with (
        InputFolderLock(input_folder) as input_lock,
        pytest.raises(PermissionError, match="injected cache write"),
    ):
        prepare_processing_cache(
            cache_folder,
            input_lock=input_lock,
            reset_cache=False,
        )
    assert legacy_artifact.read_text(encoding="utf-8") == "keep"


def test_reset_cache_removes_only_processing_cache_root(tmp_path: Path) -> None:
    """reset actionでprocessing cache rootだけが再作成されること。

    Arrange:
        - processing cache、model store、Output Folderが用意される
    Act:
        - reset指定でprocessing cacheが準備される
    Assert:
        - cache内容だけが消えmodel storeとoutputが保持されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    cache_folder = input_folder / ".game-screen-pick" / "cache"
    cached_artifact = cache_folder / "videos" / "fingerprint" / "artifact.json"
    cached_artifact.parent.mkdir(parents=True)
    cached_artifact.write_text("cached", encoding="utf-8")
    model_store = tmp_path / "model-store"
    model_store.mkdir()
    (model_store / "model.bin").write_text("model", encoding="utf-8")
    output_folder = tmp_path / "output"
    output_folder.mkdir()
    (output_folder / "keep.webp").write_text("output", encoding="utf-8")

    # Act
    with InputFolderLock(input_folder) as input_lock:
        diagnostic = prepare_processing_cache(
            cache_folder,
            input_lock=input_lock,
            reset_cache=True,
        )

    # Assert
    assert diagnostic.removed_entry_count == 0
    assert diagnostic.removed_bytes == 0
    assert cache_folder.is_dir()
    assert tuple(cache_folder.iterdir()) == ()
    assert (model_store / "model.bin").read_text(encoding="utf-8") == "model"
    assert (output_folder / "keep.webp").read_text(encoding="utf-8") == "output"


def test_cache_preparation_requires_the_matching_active_input_lock(
    tmp_path: Path,
) -> None:
    """processing cache mutationに対応する保持中Input Lockが要求されること。

    Arrange:
        - lock未取得のVideo Input Folderが用意される
    Act:
        - processing cache preparationが直接実行される
    Assert:
        - mutation前にInput Lock errorとなること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    input_lock = InputFolderLock(input_folder)
    cache_folder = input_folder / ".game-screen-pick" / "cache"

    # Act
    # Assert
    with pytest.raises(RuntimeError, match="Input Lock"):
        prepare_processing_cache(
            cache_folder,
            input_lock=input_lock,
            reset_cache=False,
        )
    assert not cache_folder.exists()
