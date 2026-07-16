import pytest

from src.video_selection.models.collected_context import CollectedContext
from src.video_selection.models.context_cue import ContextCue


def test_context_cues_and_optional_speech_runtime_are_separated() -> None:
    """Context CueとSTT実行有無が一つの収集結果として保持されること。

    Arrange:
        - 一つのContext CueとSpeech Runtime Identityが用意される
    Act:
        - Collected Contextが構築される
    Assert:
        - CueとSTT実行identityが保持されること
    """
    # Arrange
    cue = ContextCue(identifier="cue-001")

    # Act
    collected = CollectedContext(
        cues=(cue,),
        speech_runtime_identity="speech_runtime:v1",
    )

    # Assert
    assert collected.cues == (cue,)
    assert collected.speech_runtime_identity == "speech_runtime:v1"


def test_empty_speech_runtime_identity_is_rejected() -> None:
    """空のSpeech Runtime IdentityがSTT実行済みとして扱われないこと。

    Arrange:
        - 空のSpeech Runtime Identityが用意される
    Act:
        - Collected Contextの構築が試行される
    Assert:
        - validation errorになること
    """
    # Arrange
    identity = ""

    # Act
    # Assert
    with pytest.raises(ValueError, match="Speech Runtime Identity"):
        CollectedContext(cues=(), speech_runtime_identity=identity)
