"""言語とdefault dispositionによるContext stream選択。"""

from collections.abc import Iterable

from ..models.context_stage_error import ContextStageError
from ..models.context_stage_failure_reason import ContextStageFailureReason
from ..models.media_stream import MediaStream

_LANGUAGE_ALIASES = {"jpn": "ja", "eng": "en"}


def select_unique_context_stream(
    candidates: Iterable[MediaStream],
    language: str,
    ambiguous_reason: ContextStageFailureReason,
) -> MediaStream | None:
    """設定言語とdefault dispositionから一意なstreamを返す。"""
    candidate_tuple = tuple(candidates)
    normalized_language = _normalize_language(language)
    matching = tuple(
        stream
        for stream in candidate_tuple
        if _normalize_language(stream.language) == normalized_language
    )
    eligible = matching or tuple(
        stream
        for stream in candidate_tuple
        if _normalize_language(stream.language) is None
    )
    defaults = tuple(stream for stream in eligible if stream.is_default)
    if len(defaults) == 1:
        return defaults[0]
    if len(defaults) > 1 or len(eligible) > 1:
        raise ContextStageError(ambiguous_reason)
    return eligible[0] if eligible else None


def _normalize_language(language: str | None) -> str | None:
    if language is None:
        return None
    primary = language.strip().lower().replace("_", "-").split("-", maxsplit=1)[0]
    if not primary or primary == "und":
        return None
    return _LANGUAGE_ALIASES.get(primary, primary)
