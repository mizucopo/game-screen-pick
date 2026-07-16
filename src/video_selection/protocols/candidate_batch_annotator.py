"""walking skeletonの一括Candidate Annotation port。"""

from typing import Protocol

from ..models.candidate_annotation import CandidateAnnotation
from ..models.context_cue import ContextCue
from ..models.frame_candidate import FrameCandidate
from ..models.resolved_model import ResolvedModel


class CandidateBatchAnnotator(Protocol):
    """Issue #190までのwalking skeleton用移行seam。"""

    def annotate_candidates(
        self,
        candidates: tuple[FrameCandidate, ...],
        context_cues: tuple[ContextCue, ...],
        model: ResolvedModel,
    ) -> tuple[CandidateAnnotation, ...]:
        """候補に対応するplaceholder annotationを返す。"""
