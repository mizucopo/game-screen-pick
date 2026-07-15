"""Composite Video Scanが生成した一つのproxy frame。"""

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


@dataclass(frozen=True)
class ScannedVideoFrame:
    """exact source timingと一時または永続JPEG pathを保持する。"""

    source_pts: int
    duration_ts: int | None
    time_base: Fraction
    width: int
    height: int
    image_path: Path

    def __post_init__(self) -> None:
        """time baseとproxy寸法を検証する。"""
        if self.time_base <= 0 or self.width < 1 or self.height < 1:
            msg = "Scanned Video Frameには正のtime baseと寸法が必要です"
            raise ValueError(msg)
