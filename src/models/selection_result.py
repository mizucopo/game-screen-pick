"""選定結果。"""

from dataclasses import dataclass, field
from typing import Generic, TypeVar

from .selection_annotation import SelectionAnnotation

SelectionCandidateT = TypeVar("SelectionCandidateT")


@dataclass(frozen=True)
class SelectionResult(Generic[SelectionCandidateT]):
    """選定で得られる結果一式."""

    selected: list[SelectionCandidateT]
    rejected_by_similarity: int
    target_counts: dict[str, int]
    actual_counts: dict[str, int]
    annotations_by_path: dict[str, SelectionAnnotation] = field(default_factory=dict)

    def annotation_for(self, path: str) -> SelectionAnnotation:
        """候補に対応する選定注釈を返す."""
        return self.annotations_by_path.get(path, SelectionAnnotation())
