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
    assert version == "context-collection-v1"


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


def test_select_images_keeps_walking_skeleton_version_until_selector_is_wired() -> None:
    """旧selectorの実行中はwalking skeleton versionが維持されること。

    Arrange:
        - select-images Processing Stageが用意される
    Act:
        - Stage versionが解決される
    Assert:
        - 現行のwalking skeleton versionが返されること
    """
    # Arrange
    stage = ProcessingStage.SELECT_IMAGES

    # Act
    version = stage_version(stage)

    # Assert
    assert version == "walking-skeleton-0"
