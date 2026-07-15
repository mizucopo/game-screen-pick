"""walking skeletonのCandidate Annotation。"""

from dataclasses import dataclass

from .frame_candidate import FrameCandidate


@dataclass(frozen=True)
class CandidateAnnotation:
    """fake VisionRuntimeが返す最小annotation。"""

    candidate: FrameCandidate
    summary: str
