"""Video Scan Stageのscene signal metadata。"""

import math
from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class SceneSignal:
    """永続画像を持たないexact scene signal。"""

    source_pts: int
    video_time: Fraction
    quality_score: float
    eligible: bool

    def __post_init__(self) -> None:
        """timeline位置と画質値を検証する。"""
        if self.video_time < 0 or not math.isfinite(self.quality_score):
            msg = "Scene Signalには有効な時刻と画質値が必要です"
            raise ValueError(msg)
