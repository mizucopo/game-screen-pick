"""MediaRuntimeの一回のnative Video Scan結果。"""

import math
from dataclasses import dataclass
from fractions import Fraction

from .scanned_video_frame import ScannedVideoFrame


@dataclass(frozen=True)
class NativeVideoScan:
    """timeline端点、heartbeat、scene、一回のdecode metricを保持する。"""

    stream_index: int
    origin_pts: int
    last_frame_pts: int
    last_frame_duration_ts: int | None
    time_base: Fraction
    heartbeats: tuple[ScannedVideoFrame, ...]
    scene_frames: tuple[ScannedVideoFrame, ...]
    wall_seconds: float
    cpu_seconds: float
    decode_pass_count: int

    def __post_init__(self) -> None:
        """scanが一回のdecodeと1件以上のheartbeatを持つことを検証する。"""
        if (
            self.time_base <= 0
            or not self.heartbeats
            or self.decode_pass_count != 1
            or not math.isfinite(self.wall_seconds)
            or not math.isfinite(self.cpu_seconds)
            or self.wall_seconds < 0
            or self.cpu_seconds < 0
        ):
            msg = "Native Video Scanのtimingまたはmetricが不正です"
            raise ValueError(msg)
