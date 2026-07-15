"""MediaRuntimeのsemantic port。"""

from typing import Protocol

from ..models.frame_candidate import FrameCandidate
from ..models.video_set import VideoSet


class MediaRuntime(Protocol):
    """Video SetからFrame Candidateを取得する境界。"""

    def extract_candidates(
        self,
        video_set: VideoSet,
    ) -> tuple[FrameCandidate, ...]:
        """Video SetのFrame Candidateを返す。"""
