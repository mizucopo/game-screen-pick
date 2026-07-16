"""同じ内容を示すContext Cueの関連。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ContextCueEquivalenceGroup:
    """provenanceを残したまま二重annotationを防ぐCue集合。"""

    representative_cue_id: str
    cue_ids: tuple[str, ...]
