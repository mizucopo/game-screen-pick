"""walking skeletonのFrame Candidate抽出placeholder port。"""

from typing import Protocol

from ..models.frame_candidate import FrameCandidate
from ..models.video_set import VideoSet


class FrameCandidateExtractor(Protocol):
    """Issue #182でtimeline serviceへ置換される候補抽出境界。"""

    def extract_candidates(
        self,
        video_set: VideoSet,
    ) -> tuple[FrameCandidate, ...]:
        """Video SetのFrame Candidateを返す。"""
