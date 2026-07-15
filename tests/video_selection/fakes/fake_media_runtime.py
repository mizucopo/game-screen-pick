from src.video_selection.models.frame_candidate import FrameCandidate
from src.video_selection.models.video_set import VideoSet


class FakeMediaRuntime:
    """固定されたFrame Candidateを返すfake。"""

    def __init__(self, candidates: tuple[FrameCandidate, ...]) -> None:
        self._candidates = candidates

    def extract_candidates(
        self,
        video_set: VideoSet,
    ) -> tuple[FrameCandidate, ...]:
        """Video Setに対する候補を返す。"""
        del video_set
        return self._candidates
