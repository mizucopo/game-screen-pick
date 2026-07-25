"""選定前fingerprint用Select Images cache indexのtest。"""

from pathlib import Path

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
