"""Video Scan StageのHeartbeat Proxy。"""

import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


@dataclass(frozen=True)
class HeartbeatProxy:
    """exact時刻、cache path、model-free画質評価を持つproxy。"""

    source_pts: int
    video_time: Fraction
    proxy_path: Path
    quality_score: float
    eligible: bool

    def __post_init__(self) -> None:
        """timeline位置と画質値を検証する。"""
        if self.video_time < 0 or not math.isfinite(self.quality_score):
            msg = "Heartbeat Proxyには有効な時刻と画質値が必要です"
            raise ValueError(msg)
