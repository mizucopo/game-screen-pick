"""forced subtitleとSTT Context Cueの等価group構築。"""

import unicodedata

from ..models.context_cue import ContextCue
from ..models.context_cue_equivalence_group import ContextCueEquivalenceGroup


def build_context_cue_equivalence_groups(
    cues: tuple[ContextCue, ...],
) -> tuple[ContextCueEquivalenceGroup, ...]:
    """正規化本文が一致し時間が重なるcross-source Cueをまとめる。"""
    parents = list(range(len(cues)))
    for left_index, left in enumerate(cues):
        for right_index in range(left_index + 1, len(cues)):
            right = cues[right_index]
            if (
                left.source_kind != right.source_kind
                and _comparison_text(left.text) == _comparison_text(right.text)
                and left.start < right.end
                and right.start < left.end
            ):
                _union(parents, left_index, right_index)
    components: dict[int, list[int]] = {}
    for index in range(len(cues)):
        components.setdefault(_find(parents, index), []).append(index)
    groups: list[ContextCueEquivalenceGroup] = []
    for indexes in components.values():
        if len(indexes) < 2:
            continue
        representative_index = min(
            indexes,
            key=lambda index: (
                cues[index].source_kind != "embedded_subtitle",
                cues[index].start,
                cues[index].identifier,
            ),
        )
        groups.append(
            ContextCueEquivalenceGroup(
                representative_cue_id=cues[representative_index].identifier,
                cue_ids=tuple(cues[index].identifier for index in indexes),
            )
        )
    return tuple(groups)


def _comparison_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(
        character
        for character in normalized
        if not character.isspace()
        and not unicodedata.category(character).startswith("P")
    )


def _find(parents: list[int], index: int) -> int:
    while parents[index] != index:
        parents[index] = parents[parents[index]]
        index = parents[index]
    return index


def _union(parents: list[int], left: int, right: int) -> None:
    left_root = _find(parents, left)
    right_root = _find(parents, right)
    if left_root != right_root:
        parents[right_root] = left_root
