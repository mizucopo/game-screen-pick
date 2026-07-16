from src.video_selection.models.candidate_annotation import CandidateAnnotation
from src.video_selection.models.context_cue import ContextCue
from src.video_selection.models.frame_candidate import FrameCandidate
from src.video_selection.models.resolved_model import ResolvedModel


class FailingVisionRuntime:
    """Candidate Annotation Stageを失敗させるfake。"""

    def annotate_candidates(
        self,
        candidates: tuple[FrameCandidate, ...],
        context_cues: tuple[ContextCue, ...],
        model: ResolvedModel,
    ) -> tuple[CandidateAnnotation, ...]:
        """外部runtime failureを送出する。"""
        del candidates, context_cues, model
        msg = "fake vision failure"
        raise RuntimeError(msg)
