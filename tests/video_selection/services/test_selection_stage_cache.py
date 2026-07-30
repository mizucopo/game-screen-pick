"""選定前fingerprint用Select Images cache indexのtest。"""

from pathlib import Path

import pytest

from src.video_selection.models.processing_stage import ProcessingStage
from src.video_selection.models.stage_fingerprint import StageFingerprint
from src.video_selection.services.build_stage_fingerprint import (
    build_stage_fingerprint,
)
from src.video_selection.services.completed_stage_writer import CompletedStageWriter
from src.video_selection.services.selection_stage_cache import SelectionStageCache


def test_request_fingerprint_restores_verified_completed_selection(
    tmp_path: Path,
) -> None:
    """選定前fingerprintからintegrity検証済みSelect Imagesが復元されること。

    Arrange:
        - 確定済みSelect Images Stageと対応するrequest indexが用意される
    Act:
        - selectorを実行する前にrequest fingerprintでcacheが検索される
    Assert:
        - artifactと完全なCompleted Stage identityが返されること
    """
    # Arrange
    video_set_fingerprint = "a" * 64
    request_fingerprint = StageFingerprint("b" * 64)
    upstream = (StageFingerprint("c" * 64),)
    semantic_input = {
        "selection_request_fingerprint": request_fingerprint.value,
        "requested_count": 1,
    }
    stage_fingerprint = build_stage_fingerprint(
        ProcessingStage.SELECT_IMAGES,
        upstream,
        semantic_input,
    )
    completed = CompletedStageWriter(
        tmp_path,
        subject_namespace="video-sets",
        subject_fingerprint=video_set_fingerprint,
    ).write(
        ProcessingStage.SELECT_IMAGES,
        stage_fingerprint,
        upstream,
        semantic_input,
        {"schema": "selection-test"},
    )
    cache = SelectionStageCache(
        tmp_path,
        video_set_fingerprint=video_set_fingerprint,
    )
    cache.record(request_fingerprint, completed)

    # Act
    restored = cache.read(request_fingerprint)

    # Assert
    assert restored is not None
    artifact, restored_completed = restored
    assert artifact == {"schema": "selection-test"}
    assert restored_completed == completed


def test_missing_request_index_is_rebuilt_from_completed_selection(
    tmp_path: Path,
) -> None:
    """欠落したrequest indexが確定済みSelect Imagesから再構築されること。

    Arrange:
        - index未保存のままatomic確定されたSelect Images Stageが用意される
    Act:
        - 選定前request fingerprintでcacheが検索される
    Assert:
        - 検証済みartifactが復元されrequest indexも再構築されること
    """
    # Arrange
    video_set_fingerprint = "a" * 64
    request_fingerprint = StageFingerprint("b" * 64)
    upstream = (StageFingerprint("c" * 64),)
    semantic_input = {
        "selection_request_fingerprint": request_fingerprint.value,
        "requested_count": 1,
    }
    stage_fingerprint = build_stage_fingerprint(
        ProcessingStage.SELECT_IMAGES,
        upstream,
        semantic_input,
    )
    completed = CompletedStageWriter(
        tmp_path,
        subject_namespace="video-sets",
        subject_fingerprint=video_set_fingerprint,
    ).write(
        ProcessingStage.SELECT_IMAGES,
        stage_fingerprint,
        upstream,
        semantic_input,
        {"schema": "selection-test"},
    )
    cache = SelectionStageCache(
        tmp_path,
        video_set_fingerprint=video_set_fingerprint,
    )

    # Act
    restored = cache.read(request_fingerprint)

    # Assert
    assert restored == ({"schema": "selection-test"}, completed)
    index_path = (
        tmp_path
        / ".indexes"
        / "video-sets"
        / video_set_fingerprint
        / ProcessingStage.SELECT_IMAGES.value
        / f"{request_fingerprint.value}.json"
    )
    assert index_path.is_file()


def test_index_permission_failure_does_not_trigger_recovery_or_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """indexのaccess障害がcache missや再構築へ変換されないこと。

    Arrange:
        - 確定済みSelection Stageと対応indexが用意される
        - indexの読込だけがPermissionErrorになる
    Act:
        - request fingerprintでcacheが検索される
    Assert:
        - access障害が返されindex bytesが保持されること
    """
    # Arrange
    video_set_fingerprint = "d" * 64
    request_fingerprint = StageFingerprint("e" * 64)
    upstream = (StageFingerprint("f" * 64),)
    semantic_input = {
        "selection_request_fingerprint": request_fingerprint.value,
        "requested_count": 1,
    }
    stage_fingerprint = build_stage_fingerprint(
        ProcessingStage.SELECT_IMAGES,
        upstream,
        semantic_input,
    )
    completed = CompletedStageWriter(
        tmp_path,
        subject_namespace="video-sets",
        subject_fingerprint=video_set_fingerprint,
    ).write(
        ProcessingStage.SELECT_IMAGES,
        stage_fingerprint,
        upstream,
        semantic_input,
        {"schema": "selection-test"},
    )
    cache = SelectionStageCache(
        tmp_path,
        video_set_fingerprint=video_set_fingerprint,
    )
    cache.record(request_fingerprint, completed)
    index_path = (
        tmp_path
        / ".indexes"
        / "video-sets"
        / video_set_fingerprint
        / ProcessingStage.SELECT_IMAGES.value
        / f"{request_fingerprint.value}.json"
    )
    original_read_text = Path.read_text
    original_bytes = index_path.read_bytes()

    def deny_index_read(
        path: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        if path == index_path:
            raise PermissionError("injected index permission failure")
        return original_read_text(path, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "read_text", deny_index_read)

    # Act / Assert
    with pytest.raises(PermissionError, match="injected index permission failure"):
        cache.read(request_fingerprint)
    assert index_path.read_bytes() == original_bytes


@pytest.mark.parametrize("operation", ("read", "record", "discard"))
def test_symlinked_index_ancestor_is_rejected_before_cache_access(
    tmp_path: Path,
    operation: str,
) -> None:
    """symlink化されたindex祖先を経由するcache操作が拒否されること。

    Arrange:
        - 確定済みSelect Images Stageと外部directoryを指す`.indexes`が用意される
    Act:
        - request indexのread、record、discardのいずれかが実行される
    Assert:
        - symlinkとして拒否され外部directoryとCompleted Stageが変更されないこと
    """
    # Arrange
    video_set_fingerprint = "1" * 64
    request_fingerprint = StageFingerprint("2" * 64)
    upstream = (StageFingerprint("3" * 64),)
    semantic_input = {
        "selection_request_fingerprint": request_fingerprint.value,
        "requested_count": 1,
    }
    stage_fingerprint = build_stage_fingerprint(
        ProcessingStage.SELECT_IMAGES,
        upstream,
        semantic_input,
    )
    writer = CompletedStageWriter(
        tmp_path,
        subject_namespace="video-sets",
        subject_fingerprint=video_set_fingerprint,
    )
    completed = writer.write(
        ProcessingStage.SELECT_IMAGES,
        stage_fingerprint,
        upstream,
        semantic_input,
        {"schema": "selection-test"},
    )
    external = tmp_path.parent / f"{tmp_path.name}-external-index"
    external.mkdir()
    (tmp_path / ".indexes").symlink_to(external, target_is_directory=True)
    cache = SelectionStageCache(
        tmp_path,
        video_set_fingerprint=video_set_fingerprint,
    )

    # Act
    with pytest.raises(ValueError, match="symbolic link"):
        if operation == "read":
            cache.read(request_fingerprint)
        elif operation == "record":
            cache.record(request_fingerprint, completed)
        else:
            cache.discard(request_fingerprint, completed)

    # Assert
    assert tuple(external.iterdir()) == ()
    assert writer.read(
        ProcessingStage.SELECT_IMAGES,
        stage_fingerprint,
        upstream,
        semantic_input,
    ) == {"schema": "selection-test"}
