"""Context用embedded subtitle stream選択。"""

from ..models.context_stage_error import ContextStageError
from ..models.context_stage_failure_reason import ContextStageFailureReason
from ..models.media_probe import MediaProbe
from ..models.media_stream import MediaStream
from .select_unique_context_stream import select_unique_context_stream

_TEXT_SUBTITLE_CODECS = frozenset(
    {
        "ass",
        "jacosub",
        "microdvd",
        "mov_text",
        "sami",
        "ssa",
        "subrip",
        "text",
        "webvtt",
    }
)


def select_context_subtitle_stream(
    probe: MediaProbe,
    language: str,
    explicit_index: int | None,
) -> MediaStream | None:
    """non-forced text subtitleを優先して一意なstreamを返す。"""
    subtitle_streams = tuple(
        stream for stream in probe.streams if stream.kind == "subtitle"
    )
    if explicit_index is not None:
        selected = next(
            (stream for stream in subtitle_streams if stream.index == explicit_index),
            None,
        )
        if selected is None:
            raise ContextStageError(ContextStageFailureReason.INVALID_SUBTITLE_STREAM)
        if selected.codec_name not in _TEXT_SUBTITLE_CODECS:
            raise ContextStageError(
                ContextStageFailureReason.UNSUPPORTED_BITMAP_SUBTITLE
            )
        return selected
    text_streams = tuple(
        stream
        for stream in subtitle_streams
        if stream.codec_name in _TEXT_SUBTITLE_CODECS
    )
    for candidates in (
        tuple(stream for stream in text_streams if not stream.is_forced),
        tuple(stream for stream in text_streams if stream.is_forced),
    ):
        selected = select_unique_context_stream(
            candidates,
            language,
            ContextStageFailureReason.AMBIGUOUS_SUBTITLE_STREAM,
        )
        if selected is not None:
            return selected
    unsupported_streams = tuple(
        stream
        for stream in subtitle_streams
        if stream.codec_name not in _TEXT_SUBTITLE_CODECS
    )
    for candidates in (
        tuple(stream for stream in unsupported_streams if not stream.is_forced),
        tuple(stream for stream in unsupported_streams if stream.is_forced),
    ):
        selected = select_unique_context_stream(
            candidates,
            language,
            ContextStageFailureReason.AMBIGUOUS_SUBTITLE_STREAM,
        )
        if selected is not None:
            raise ContextStageError(
                ContextStageFailureReason.UNSUPPORTED_BITMAP_SUBTITLE
            )
    return None
