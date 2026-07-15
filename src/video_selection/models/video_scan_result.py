"""Completed Video Scan Stageのdomain result。"""

from dataclasses import dataclass

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
