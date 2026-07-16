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
