"""scene mix 選定結果。"""

from dataclasses import dataclass, field
from typing import Generic, TypeVar

from .scene_mix_candidate import SceneMixCandidate
from .selection_annotation import SelectionAnnotation

SceneMixCandidateT = TypeVar("SceneMixCandidateT", bound=SceneMixCandidate)


@dataclass(frozen=True)
class SelectionResult(Generic[SceneMixCandidateT]):
    """scene mix 選定で得られる結果一式."""

    selected: list[SceneMixCandidateT]
    rejected_by_similarity: int
    target_counts: dict[str, int]
    actual_counts: dict[str, int]
    annotations_by_path: dict[str, SelectionAnnotation] = field(default_factory=dict)

    def annotation_for(self, candidate: SceneMixCandidate) -> SelectionAnnotation:
        """候補に対応する選定注釈を返す."""
        return self.annotations_by_path.get(candidate.path, SelectionAnnotation())
