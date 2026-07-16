"""forced subtitleとSTT Context Cueの等価group構築。"""

import unicodedata
from fractions import Fraction

from ..models.context_cue import ContextCue
from ..models.context_cue_equivalence_group import ContextCueEquivalenceGroup


def build_context_cue_equivalence_groups(
    cues: tuple[ContextCue, ...],
) -> tuple[ContextCueEquivalenceGroup, ...]:
    """同じ発話occurrenceを表すsubtitle/STT Cueを一対一でまとめる。"""
    subtitle_indexes = tuple(
        index
        for index, cue in enumerate(cues)
        if cue.source_kind == "embedded_subtitle"
    )
    speech_indexes = tuple(
        index for index, cue in enumerate(cues) if cue.source_kind == "speech_to_text"
    )
    candidates = sorted(
        (
            (subtitle_index, speech_index)
            for subtitle_index in subtitle_indexes
            for speech_index in speech_indexes
            if _are_equivalent(cues[subtitle_index], cues[speech_index])
        ),
        key=lambda pair: _pair_priority(cues[pair[0]], cues[pair[1]]),
    )
    used_subtitles: set[int] = set()
    used_speech: set[int] = set()
    selected_pairs: list[tuple[int, int]] = []
    for subtitle_index, speech_index in candidates:
        if subtitle_index in used_subtitles or speech_index in used_speech:
            continue
        used_subtitles.add(subtitle_index)
        used_speech.add(speech_index)
        selected_pairs.append((subtitle_index, speech_index))
    selected_pairs.sort(
        key=lambda pair: (
            cues[pair[0]].start,
            cues[pair[0]].identifier,
            cues[pair[1]].identifier,
        )
    )
    return tuple(
        ContextCueEquivalenceGroup(
            representative_cue_id=cues[subtitle_index].identifier,
            cue_ids=(
                cues[subtitle_index].identifier,
                cues[speech_index].identifier,
            ),
        )
        for subtitle_index, speech_index in selected_pairs
    )


def _are_equivalent(subtitle: ContextCue, speech: ContextCue) -> bool:
    return (
        _comparison_text(subtitle.text) == _comparison_text(speech.text)
        and subtitle.start < speech.end
        and speech.start < subtitle.end
    )


def _pair_priority(
    subtitle: ContextCue,
    speech: ContextCue,
) -> tuple[Fraction, Fraction, Fraction, Fraction, str, str]:
    overlap = min(subtitle.end, speech.end) - max(subtitle.start, speech.start)
    midpoint_distance = abs(subtitle.start + subtitle.end - speech.start - speech.end)
    return (
        -overlap,
        midpoint_distance,
        subtitle.start,
        speech.start,
        subtitle.identifier,
        speech.identifier,
    )


def _comparison_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(
        character
        for character in normalized
        if not character.isspace()
        and not unicodedata.category(character).startswith("P")
    )
