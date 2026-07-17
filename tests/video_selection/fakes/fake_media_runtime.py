from collections.abc import Callable

from src.video_selection.models.frame_candidate import FrameCandidate
from src.video_selection.models.video_set import VideoSet


class FakeMediaRuntime:
    """固定されたFrame Candidateを返すfake。"""

    def __init__(
        self,
        candidates: tuple[FrameCandidate, ...],
        *,
        on_extract_candidates: Callable[[], None] | None = None,
    ) -> None:
        self._candidates = candidates
        self._on_extract_candidates = on_extract_candidates

    def extract_candidates(
        self,
        video_set: VideoSet,
    ) -> tuple[FrameCandidate, ...]:
        """Video Setに対する候補を返す。"""
        del video_set
        if self._on_extract_candidates is not None:
            self._on_extract_candidates()
        return self._candidates
