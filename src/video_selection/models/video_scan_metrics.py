"""Video Scan Stageの比較可能metric。"""

from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class VideoScanMetrics:
    """exact duration、処理時間、proxy coverageと容量を保持する。"""

    input_duration: Fraction
    wall_seconds: float
    cpu_seconds: float
    input_seconds_per_wall_second: float
    decode_backend: str
    decode_pass_count: int
    heartbeat_count: int
    heartbeat_bytes: int
    heartbeat_max_gap_seconds: float
    heartbeat_p95_gap_seconds: float
    scene_signal_count: int
    timeline_segment_count: int
