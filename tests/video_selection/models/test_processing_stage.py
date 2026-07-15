"""walking skeletonのProcessing Stage契約test。"""

from src.video_selection.models.processing_stage import ProcessingStage


def test_atomic_publication_is_outside_reusable_processing_stages() -> None:
    """Output Folder publicationが再利用可能Stageに含まれないこと。

    Arrange:
        - walking skeletonのProcessing Stage列が用意される
    Act:
        - Stage valueが列挙される
    Assert:
        - atomic renameによるpublicationがCompleted Stageでないこと
    """
    # Arrange
    stages = tuple(ProcessingStage)

    # Act
    stage_values = tuple(stage.value for stage in stages)

    # Assert
    assert "publish-output" not in stage_values
