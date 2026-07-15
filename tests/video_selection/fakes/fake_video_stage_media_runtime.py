"""Video Stage processor test用MediaRuntime fake。"""

from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from src.video_selection.models.decoded_video_frame import DecodedVideoFrame
from src.video_selection.models.media_probe import MediaProbe
from src.video_selection.models.media_runtime_identity import MediaRuntimeIdentity
from src.video_selection.models.media_stream import MediaStream
from src.video_selection.models.native_video_scan import NativeVideoScan
from src.video_selection.models.scanned_video_frame import ScannedVideoFrame


class FakeVideoStageMediaRuntime:
    """決定的なscanとnative frameを返し呼び出し順を記録するfake。"""

    def __init__(self) -> None:
        self.scan_calls: list[Path] = []
        self.range_calls: list[Path] = []
        self.call_order: list[tuple[str, str]] = []

    def preflight(self) -> MediaRuntimeIdentity:
        """固定runtime identityを返す。"""
        return MediaRuntimeIdentity("6.1.1-test", "6.1.1-test")

    def probe(self, media_path: Path) -> MediaProbe:
        """一つのdefault video streamを返す。"""
        self.call_order.append(("probe", media_path.name))
        return MediaProbe(
            format_names=("matroska",),
            streams=(
                MediaStream(
                    index=0,
                    kind="video",
                    codec_name="ffv1",
                    time_base=Fraction(1, 10),
                    start_pts=0,
                    duration_ts=20,
                    width=64,
                    height=48,
                    sample_rate=None,
                    channels=None,
                    language=None,
                    is_default=True,
                    is_forced=False,
                    is_attached_picture=False,
                ),
            ),
        )

    def scan_video(
        self,
        media_path: Path,
        stream: MediaStream,
        artifact_folder: Path,
        *,
        heartbeat_interval_seconds: float,
        scene_change_threshold: float,
        scene_min_interval_seconds: float,
        decode_backend: str,
    ) -> NativeVideoScan:
        """2件のheartbeatと1件の一時scene frameを生成する。"""
        del (
            heartbeat_interval_seconds,
            scene_change_threshold,
            scene_min_interval_seconds,
            decode_backend,
        )
        self.scan_calls.append(media_path)
        self.call_order.append(("scan", media_path.name))
        heartbeat_folder = artifact_folder / "heartbeats"
        scene_folder = artifact_folder / ".scene-proxies"
        heartbeat_folder.mkdir(parents=True)
        scene_folder.mkdir()
        heartbeats = tuple(
            self._write_scan_frame(heartbeat_folder / f"{pts:012d}.jpg", pts)
            for pts in (0, 10)
        )
        scenes = (self._write_scan_frame(scene_folder / "000000000010.jpg", 10),)
        return NativeVideoScan(
            stream_index=stream.index,
            origin_pts=0,
            last_frame_pts=10,
            last_frame_duration_ts=10,
            time_base=Fraction(1, 10),
            heartbeats=heartbeats,
            scene_frames=scenes,
            wall_seconds=0.1,
            cpu_seconds=0.05,
            decode_pass_count=1,
        )

    def scan_video_frame_ranges(
        self,
        media_path: Path,
        stream_index: int,
        pts_ranges: tuple[tuple[int, int], ...],
        max_dimension: int,
    ) -> Iterator[DecodedVideoFrame]:
        """range内のnative test frameを返す。"""
        del max_dimension
        self.range_calls.append(media_path)
        self.call_order.append(("refine", media_path.name))
        for pts in (0, 5, 10, 15):
            if any(start <= pts < end for start, end in pts_ranges):
                yield self._decoded_frame(stream_index, pts)

    def write_mjpeg_proxy(
        self,
        frame: DecodedVideoFrame,
        output_path: Path,
        *,
        quality: int,
    ) -> None:
        """test用JPEG proxyを保存する。"""
        del quality
        rgb = np.frombuffer(frame.pixels, dtype=np.uint8).reshape(
            frame.height,
            frame.width,
            3,
        )
        success, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if not success:
            raise RuntimeError("test JPEG encode failed")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(encoded.tobytes())

    def _write_scan_frame(self, path: Path, pts: int) -> ScannedVideoFrame:
        frame = self._decoded_frame(0, pts)
        self.write_mjpeg_proxy(frame, path, quality=3)
        return ScannedVideoFrame(
            source_pts=pts,
            duration_ts=5,
            time_base=Fraction(1, 10),
            width=frame.width,
            height=frame.height,
            image_path=path,
        )

    @staticmethod
    def _decoded_frame(stream_index: int, pts: int) -> DecodedVideoFrame:
        rows, columns = np.indices((48, 64))
        values = ((rows // 3 + columns // 4 + pts // 5) % 3 * 90 + 25).astype(np.uint8)
        rgb = np.stack((values, np.roll(values, 5, axis=1), 255 - values), axis=2)
        return DecodedVideoFrame(
            stream_index=stream_index,
            pts=pts,
            duration_ts=5,
            time_base=Fraction(1, 10),
            width=64,
            height=48,
            pixel_format="rgb24",
            pixels=rgb.tobytes(),
        )
