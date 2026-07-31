"""CheckpointOperationの契約test。"""

from src.video_selection.models.checkpoint_operation import CheckpointOperation


def test_checkpoint_operations_are_explicit_and_unique() -> None:
    """全durable checkpoint種別が明示され値が重複しないこと。

    Arrange:
        - 現行pipelineが持つ最小checkpoint種別が用意される
    Act:
        - enum値が列挙される
    Assert:
        - 必要な全種別が一意なstable codeとして返されること
    """
    # Arrange
    expected = {
        "video-identity",
        "video-scan-partition",
        "frame-refinement-group",
        "pcm-audio-chunk",
        "speech-recognition-chunk",
        "embedded-subtitle-stream",
        "selected-image-webp",
    }

    # Act
    actual = {operation.value for operation in CheckpointOperation}

    # Assert
    assert actual == expected
    assert len(actual) == len(tuple(CheckpointOperation))
