"""VisionRuntimeのsemantic port。"""

from typing import Protocol

from ..models.candidate_annotation import CandidateAnnotation
from ..models.context_cue import ContextCue
from ..models.frame_candidate import FrameCandidate
from ..models.resolved_model_identity import ResolvedModelIdentity


class VisionRuntime(Protocol):
    """Frame Candidateへ意味annotationを付ける境界。"""

    def annotate_candidates(
        self,
        candidates: tuple[FrameCandidate, ...],
        context_cues: tuple[ContextCue, ...],
        model_identity: ResolvedModelIdentity,
    ) -> tuple[CandidateAnnotation, ...]:
        """候補に対応するCandidate Annotationを返す。"""
