"""Completed Video Scan Stageのdomain result。"""

from dataclasses import dataclass, field

from .heartbeat_proxy import HeartbeatProxy
from .media_stream import MediaStream
from .scene_signal import SceneSignal
from .video_scan_metrics import VideoScanMetrics
from .video_timeline import VideoTimeline


@dataclass(frozen=True)
class VideoScanResult:
    """Primary stream、exact timeline、scan signal、metricを保持する。"""

    primary_stream: MediaStream
    timeline: VideoTimeline
    heartbeats: tuple[HeartbeatProxy, ...]
    scene_signals: tuple[SceneSignal, ...]
    metrics: VideoScanMetrics
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
        """resource hintが完全な正値pairであることを検証する。"""
        if (
            (self.minimum_frame_delta_ts is None)
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
        ):
            raise ValueError("Video Scanのresource hintが不正です")
