"""Processing Stage versionのtest。"""

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
    assert version == "context-collection-v3"


def test_extract_frame_candidates_has_range_seek_stage_version() -> None:
    """Frame Candidate Extractionにrange seek版が付与されること。

    Arrange:
        - extract-frame-candidates Processing Stageが用意される
    Act:
        - Stage versionが解決される
    Assert:
        - range seek実装固有のversionが返されること
    """
    # Arrange
    stage = ProcessingStage.EXTRACT_FRAME_CANDIDATES

    # Act
    version = stage_version(stage)

    # Assert
    assert version == "frame-candidate-extraction-v2"


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
    assert version == "video-set-selection-v2"
