"""Context用audio stream選択。"""

from ..models.context_stage_error import ContextStageError
from ..models.context_stage_failure_reason import ContextStageFailureReason
from ..models.media_probe import MediaProbe
from ..models.media_stream import MediaStream
from .select_unique_context_stream import select_unique_context_stream


def select_context_audio_stream(
    probe: MediaProbe,
    language: str,
    explicit_index: int | None,
) -> MediaStream | None:
    """設定言語とdefault dispositionから一意なaudio streamを返す。"""
    audio_streams = tuple(stream for stream in probe.streams if stream.kind == "audio")
    if explicit_index is not None:
        selected = next(
            (stream for stream in audio_streams if stream.index == explicit_index),
            None,
        )
        if selected is None:
            raise ContextStageError(ContextStageFailureReason.INVALID_AUDIO_STREAM)
        return selected
    return select_unique_context_stream(
        audio_streams,
        language,
        ContextStageFailureReason.AMBIGUOUS_AUDIO_STREAM,
    )
