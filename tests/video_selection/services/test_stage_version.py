"""Processing Stage versionのtest。"""

import pytest

from src.video_selection.models.processing_stage import ProcessingStage
from src.video_selection.services.stage_version import stage_version


def test_collect_context_has_context_collection_stage_version() -> None:
    """Context Collection Stageにwalking skeleton以外のversionが付与されること。

    Arrange:
        - collect-context Processing Stageが用意される
    Act:
        - Stage versionが解決される
    Assert:
        - context collection固有のversionが返されること
    """
    # Arrange
    stage = ProcessingStage.COLLECT_CONTEXT

    # Act
    version = stage_version(stage)

    # Assert
    assert version == "context-collection-v4"


def test_extract_frame_candidates_has_isolated_cpu_metric_stage_version() -> None:
    """Frame Candidate ExtractionにCPU計測分離版が付与されること。

    Arrange:
        - extract-frame-candidates Processing Stageが用意される
    Act:
        - Stage versionが解決される
    Assert:
        - CPU計測分離実装固有のversionが返されること
    """
    # Arrange
    stage = ProcessingStage.EXTRACT_FRAME_CANDIDATES

    # Act
    version = stage_version(stage)

    # Assert
    assert version == "frame-candidate-extraction-v3"


def test_scan_video_has_partition_resume_stage_version() -> None:
    """Video Scan Stageにpartition再開版のversionが付与されること。

    Arrange:
        - scan-video Processing Stageが用意される
    Act:
        - Stage versionが解決される
    Assert:
        - partition再開contract固有のversionが返されること
    """
    # Arrange
    stage = ProcessingStage.SCAN_VIDEO

    # Act
    version = stage_version(stage)

    # Assert
    assert version == "video-scan-v6"


def test_resolve_models_has_model_resolution_stage_version() -> None:
    """Model Resolution Stageに固有のcontract versionが付与されること。

    Arrange:
        - resolve-models Processing Stageが用意される
    Act:
        - Stage versionが解決される
    Assert:
        - model resolution固有のversionが返されること
    """
    # Arrange
    stage = ProcessingStage.RESOLVE_MODELS

    # Act
    version = stage_version(stage)

    # Assert
    assert version == "model-resolution-v1"


def test_select_images_uses_real_video_set_selection_version() -> None:
    """実selector接続後はVideo Set selection versionが使われること。

    Arrange:
        - select-images Processing Stageが用意される
    Act:
        - Stage versionが解決される
    Assert:
        - real Video Set selectorのversionが返されること
    """
    # Arrange
    stage = ProcessingStage.SELECT_IMAGES

    # Act
    version = stage_version(stage)

    # Assert
    assert version == "video-set-selection-v3"


@pytest.mark.parametrize("stage", tuple(ProcessingStage))
def test_every_processing_stage_has_an_explicit_version(
    stage: ProcessingStage,
) -> None:
    """全Processing Stageにsilent fallbackではないversionが登録されること。

    Arrange:
        - 定義済みの各Processing Stageが用意される
    Act:
        - Stage固有versionが解決される
    Assert:
        - 空値や共通walking-skeleton fallbackが返されないこと
    """
    # Arrange

    # Act
    version = stage_version(stage)

    # Assert
    assert version
    assert version != "walking-skeleton-0"
