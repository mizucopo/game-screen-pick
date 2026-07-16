from src.video_selection.models.candidate_annotation import CandidateAnnotation
from src.video_selection.models.context_cue import ContextCue
from src.video_selection.models.frame_candidate import FrameCandidate
from src.video_selection.models.resolved_model import ResolvedModel


class FakeVisionRuntime:
    """固定されたCandidate Annotationを返すfake。"""

    def __init__(self, annotations: tuple[CandidateAnnotation, ...]) -> None:
        self._annotations = annotations

    def annotate_candidates(
        self,
        candidates: tuple[FrameCandidate, ...],
        context_cues: tuple[ContextCue, ...],
        model: ResolvedModel,
    ) -> tuple[CandidateAnnotation, ...]:
        """候補へ付与されたannotationを返す。"""
        del candidates, context_cues, model
        return self._annotations
