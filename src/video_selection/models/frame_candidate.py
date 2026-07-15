"""walking skeletonのFrame Candidate。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class FrameCandidate:
    """fake MediaRuntimeが返す最小Frame Candidate。"""

    identifier: str
    image_bytes: bytes
