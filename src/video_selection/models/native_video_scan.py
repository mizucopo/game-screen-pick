"""MediaRuntimeの一回のnative Video Scan結果。"""

import math
from dataclasses import dataclass, field
from fractions import Fraction

from .scanned_video_frame import ScannedVideoFrame


@dataclass(frozen=True)
class NativeVideoScan:
    """timeline端点、0件以上のsignal、decode metricを保持する。"""

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
    minimum_frame_delta_ts: int | None = field(
        default=None,
        compare=False,
        repr=False,
    )
    maximum_frame_count_per_pts: int | None = field(
        default=None,
        compare=False,
        repr=False,
    )
    maximum_frame_width: int | None = field(
        default=None,
        compare=False,
        repr=False,
    )
    maximum_frame_height: int | None = field(
        default=None,
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """scanが1回以上のdecodeと有効なtimingを持つことを検証する。"""
        if (
            self.stream_index < 0
            or self.time_base <= 0
            or self.last_frame_pts < self.origin_pts
            or (
                self.last_frame_duration_ts is not None
                and self.last_frame_duration_ts <= 0
            )
            or self.decode_pass_count < 1
            or (self.minimum_frame_delta_ts is None)
            != (self.maximum_frame_count_per_pts is None)
            or (
                self.minimum_frame_delta_ts is not None
                and self.minimum_frame_delta_ts < 1
            )
            or (
                self.maximum_frame_count_per_pts is not None
                and self.maximum_frame_count_per_pts < 1
            )
            or (self.maximum_frame_width is None) != (self.maximum_frame_height is None)
            or (self.maximum_frame_width is not None and self.maximum_frame_width < 1)
            or (self.maximum_frame_height is not None and self.maximum_frame_height < 1)
            or not math.isfinite(self.wall_seconds)
            or not math.isfinite(self.cpu_seconds)
            or self.wall_seconds < 0
            or self.cpu_seconds < 0
        ):
            msg = "Native Video Scanのtimingまたはmetricが不正です"
            raise ValueError(msg)
        for frames in (self.heartbeats, self.scene_frames):
            previous_pts: int | None = None
            for frame in frames:
                if (
                    frame.time_base != self.time_base
                    or frame.source_pts < self.origin_pts
                    or frame.source_pts > self.last_frame_pts
                    or (frame.duration_ts is not None and frame.duration_ts <= 0)
                    or (previous_pts is not None and frame.source_pts <= previous_pts)
                ):
                    msg = "Native Video Scanのsignal timingが不正です"
                    raise ValueError(msg)
                previous_pts = frame.source_pts
