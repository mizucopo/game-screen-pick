"""Backgroundで先行確定されたVideo Scanのrun結果。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PreparedVideoScan:
    """cache disposition、待ち時間、cold scan速度を保持する。"""

    reused: bool
    duration_seconds: float
    input_seconds_per_wall_second: float | None
