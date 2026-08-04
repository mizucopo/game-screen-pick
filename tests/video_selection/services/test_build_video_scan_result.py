"""Video Scan result構築のtest。"""

import gc
import weakref
from fractions import Fraction
from pathlib import Path

import pytest

import src.video_selection.services.build_video_scan_result as scan_result_module
from src.video_selection.models.decoded_video_frame import DecodedVideoFrame
from src.video_selection.models.media_stream import MediaStream
from src.video_selection.models.native_video_scan import NativeVideoScan
from src.video_selection.models.scanned_video_frame import ScannedVideoFrame


def test_heartbeat_proxies_are_analyzed_without_retaining_all_decoded_rgb(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Heartbeat Proxyのdecoded RGBが全件同時保持されず解析されること。

    Arrange:
        - 複数のHeartbeat Proxyと生存frame数を記録するdecoderが用意される
    Act:
        - Native Video ScanからVideo Scan resultが構築される
    Assert:
        - 次のproxy decode時に保持される既存decoded frameが1件以下であること
        - 全Heartbeat Proxyがresultへ変換されること
    """
    # Arrange
    heartbeat_folder = tmp_path / "heartbeats"
    heartbeat_folder.mkdir()
    scanned_frames = tuple(
        ScannedVideoFrame(
            source_pts=pts,
            duration_ts=1,
            time_base=Fraction(1),
            width=2,
            height=2,
            image_path=heartbeat_folder / f"{pts:012d}.jpg",
        )
        for pts in range(8)
    )
    for frame in scanned_frames:
        frame.image_path.write_bytes(b"proxy")
    native_scan = NativeVideoScan(
        stream_index=0,
        origin_pts=0,
        last_frame_pts=7,
        last_frame_duration_ts=1,
        time_base=Fraction(1),
        heartbeats=scanned_frames,
        scene_frames=(),
        wall_seconds=1.0,
        cpu_seconds=0.5,
        decode_pass_count=1,
        minimum_frame_delta_ts=1,
        maximum_frame_count_per_pts=1,
        maximum_frame_width=1920,
        maximum_frame_height=1080,
    )
    primary_stream = MediaStream(
        index=0,
        kind="video",
        codec_name="ffv1",
        time_base=Fraction(1),
        start_pts=0,
        duration_ts=8,
        width=2,
        height=2,
        sample_rate=None,
        channels=None,
        language=None,
        is_default=True,
        is_forced=False,
        is_attached_picture=False,
    )
    frame_references: list[weakref.ReferenceType[DecodedVideoFrame]] = []
    max_retained_before_decode = 0

    def decode_proxy(
        frame: ScannedVideoFrame,
        stream_index: int,
    ) -> DecodedVideoFrame:
        nonlocal max_retained_before_decode
        gc.collect()
        max_retained_before_decode = max(
            max_retained_before_decode,
            sum(reference() is not None for reference in frame_references),
        )
        decoded = DecodedVideoFrame(
            stream_index=stream_index,
            pts=frame.source_pts,
            duration_ts=frame.duration_ts,
            time_base=frame.time_base,
            width=2,
            height=2,
            pixel_format="rgb24",
            pixels=bytes([frame.source_pts] * 12),
        )
        frame_references.append(weakref.ref(decoded))
        return decoded

    monkeypatch.setattr(scan_result_module, "_decode_proxy", decode_proxy)

    # Act
    result = scan_result_module.build_video_scan_result(
        native_scan=native_scan,
        primary_stream=primary_stream,
        video_fingerprint="d" * 64,
        decode_backend="software",
    )

    # Assert
    assert len(result.heartbeats) == len(scanned_frames)
    assert max_retained_before_decode <= 1
    assert result.minimum_frame_delta_ts == 1
    assert result.maximum_frame_count_per_pts == 1
    assert result.maximum_frame_width == 1920
    assert result.maximum_frame_height == 1080
