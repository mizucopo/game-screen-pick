"""Native Video ScanをCompleted Stage domain resultへ変換する。"""

from fractions import Fraction

import cv2
import numpy as np

from ..models.decoded_video_frame import DecodedVideoFrame
from ..models.heartbeat_proxy import HeartbeatProxy
from ..models.media_stream import MediaStream
from ..models.native_video_scan import NativeVideoScan
from ..models.scanned_video_frame import ScannedVideoFrame
from ..models.scene_signal import SceneSignal
from ..models.video_scan_metrics import VideoScanMetrics
from ..models.video_scan_result import VideoScanResult
from .analyze_neutral_images import analyze_neutral_images
from .build_exact_timeline import build_exact_timeline


def build_video_scan_result(
    *,
    native_scan: NativeVideoScan,
    primary_stream: MediaStream,
    video_fingerprint: str,
    decode_backend: str,
) -> VideoScanResult:
    """proxyを解析してexact timelineと比較可能metricを構築する。"""
    timeline = build_exact_timeline(
        video_fingerprint=video_fingerprint,
        stream=primary_stream,
        origin_pts=native_scan.origin_pts,
        last_frame_pts=native_scan.last_frame_pts,
        last_frame_duration_ts=native_scan.last_frame_duration_ts,
        scene_pts=tuple(item.source_pts for item in native_scan.scene_frames),
    )
    heartbeat_analyses = analyze_neutral_images(
        _decode_proxy(item, primary_stream.index) for item in native_scan.heartbeats
    )
    scene_analyses = analyze_neutral_images(
        _decode_proxy(item, primary_stream.index) for item in native_scan.scene_frames
    )
    heartbeats = tuple(
        HeartbeatProxy(
            source_pts=frame.source_pts,
            video_time=_video_time(frame.source_pts, native_scan),
            proxy_path=frame.image_path,
            quality_score=analysis.quality_score,
            eligible=analysis.eligible,
        )
        for frame, analysis in zip(
            native_scan.heartbeats,
            heartbeat_analyses,
            strict=True,
        )
    )
    scene_signals = tuple(
        SceneSignal(
            source_pts=frame.source_pts,
            video_time=_video_time(frame.source_pts, native_scan),
            quality_score=analysis.quality_score,
            eligible=analysis.eligible,
        )
        for frame, analysis in zip(
            native_scan.scene_frames,
            scene_analyses,
            strict=True,
        )
        if 0 <= _video_time(frame.source_pts, native_scan) < timeline.duration.seconds
    )
    gaps = [
        float(right.video_time - left.video_time)
        for left, right in zip(heartbeats, heartbeats[1:], strict=False)
    ]
    input_seconds = float(timeline.duration.seconds)
    metrics = VideoScanMetrics(
        input_duration=timeline.duration.seconds,
        wall_seconds=native_scan.wall_seconds,
        cpu_seconds=native_scan.cpu_seconds,
        input_seconds_per_wall_second=(
            input_seconds / native_scan.wall_seconds
            if native_scan.wall_seconds > 0
            else 0.0
        ),
        decode_backend=decode_backend,
        decode_pass_count=native_scan.decode_pass_count,
        heartbeat_count=len(heartbeats),
        heartbeat_bytes=sum(item.proxy_path.stat().st_size for item in heartbeats),
        heartbeat_max_gap_seconds=max(gaps, default=0.0),
        heartbeat_p95_gap_seconds=_percentile_95(gaps),
        scene_signal_count=len(scene_signals),
        timeline_segment_count=len(timeline.segments),
    )
    return VideoScanResult(
        primary_stream=primary_stream,
        timeline=timeline,
        heartbeats=heartbeats,
        scene_signals=scene_signals,
        metrics=metrics,
        minimum_frame_delta_ts=native_scan.minimum_frame_delta_ts,
        maximum_frame_count_per_pts=native_scan.maximum_frame_count_per_pts,
        maximum_frame_width=native_scan.maximum_frame_width,
        maximum_frame_height=native_scan.maximum_frame_height,
    )


def _decode_proxy(
    frame: ScannedVideoFrame,
    stream_index: int,
) -> DecodedVideoFrame:
    encoded = np.frombuffer(frame.image_path.read_bytes(), dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        msg = "Video Scan proxy画像をdecodeできません"
        raise ValueError(msg)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    return DecodedVideoFrame(
        stream_index=stream_index,
        pts=frame.source_pts,
        duration_ts=frame.duration_ts,
        time_base=frame.time_base,
        width=width,
        height=height,
        pixel_format="rgb24",
        pixels=rgb.tobytes(),
    )


def _video_time(source_pts: int, scan: NativeVideoScan) -> Fraction:
    return Fraction(source_pts - scan.origin_pts) * scan.time_base


def _percentile_95(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), 95))
