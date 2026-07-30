"""表示可能frameがなかったVideo Scan partition結果。"""

import math
from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class EmptyVideoScanPartition:
    """成功したdecodeで所有frameが0件だった半開区間を保持する。"""

    stream_index: int
    start_pts: int
    end_pts: int | None
    time_base: Fraction
    wall_seconds: float
    cpu_seconds: float
    decode_pass_count: int

    def __post_init__(self) -> None:
        """range、stream timing、decode metricを検証する。"""
        if (
            self.stream_index < 0
            or self.time_base <= 0
            or (self.end_pts is not None and self.start_pts >= self.end_pts)
            or self.decode_pass_count < 1
            or not math.isfinite(self.wall_seconds)
            or not math.isfinite(self.cpu_seconds)
            or self.wall_seconds < 0
            or self.cpu_seconds < 0
        ):
            msg = "空Video Scan partitionのrangeまたはmetricが不正です"
            raise ValueError(msg)
