"""Video Stage processor test用MediaRuntime fake。"""

import time
from collections.abc import Callable, Iterator
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from src.video_selection.models.decoded_video_frame import DecodedVideoFrame
from src.video_selection.models.embedded_subtitle import EmbeddedSubtitle
from src.video_selection.models.empty_video_scan_partition import (
    EmptyVideoScanPartition,
)
from src.video_selection.models.media_probe import MediaProbe
from src.video_selection.models.media_runtime_identity import MediaRuntimeIdentity
from src.video_selection.models.media_stream import MediaStream
from src.video_selection.models.native_video_scan import NativeVideoScan
from src.video_selection.models.pcm_audio_chunk import PcmAudioChunk
from src.video_selection.models.scanned_video_frame import ScannedVideoFrame


class FakeVideoStageMediaRuntime:
    """決定的なscanとnative frameを返し呼び出し順を記録するfake。"""

    def __init__(
        self,
        *,
        runtime_identity: MediaRuntimeIdentity | None = None,
        on_preflight: Callable[[], None] | None = None,
        on_scan_video: Callable[[Path], None] | None = None,
        on_cancel_video_scans: Callable[[], None] | None = None,
        on_scan_video_frame_ranges: Callable[[Path], None] | None = None,
        distant_moments: bool = False,
        scan_frame_pts: tuple[int, ...] | None = None,
        require_streaming_refinement: bool = False,
        cpu_burn_seconds: float = 0.0,
        reported_scan_wall_seconds: float = 0.1,
        reported_scan_cpu_seconds: float = 0.05,
        reported_refinement_child_cpu_seconds: float = 0.0,
        media_probe: MediaProbe | None = None,
        embedded_subtitles: tuple[EmbeddedSubtitle, ...] = (),
        pcm_audio_chunks: tuple[PcmAudioChunk, ...] = (),
        audio_error: Exception | None = None,
        zero_valid_frames: bool = False,
    ) -> None:
        self.scan_calls: list[Path] = []
        self.scan_partition_calls: list[tuple[Path, int, int | None]] = []
        self.range_calls: list[Path] = []
        self.range_pts_calls: list[tuple[tuple[int, int], ...]] = []
        self.call_order: list[tuple[str, str]] = []
        self._runtime_identity = runtime_identity or MediaRuntimeIdentity(
            "6.1.1-test",
            "6.1.1-test",
            "0" * 64,
        )
        self._on_preflight = on_preflight
        self._on_scan_video = on_scan_video
        self._on_cancel_video_scans = on_cancel_video_scans
        self._on_scan_video_frame_ranges = on_scan_video_frame_ranges
        self._distant_moments = distant_moments
        if scan_frame_pts is not None and (
            not scan_frame_pts or tuple(sorted(set(scan_frame_pts))) != scan_frame_pts
        ):
            raise ValueError("scan_frame_ptsは昇順で重複しない必要があります")
        self._scan_frame_pts = scan_frame_pts
        self._require_streaming_refinement = require_streaming_refinement
        self._cpu_burn_seconds = cpu_burn_seconds
        self._reported_scan_wall_seconds = reported_scan_wall_seconds
        self._reported_scan_cpu_seconds = reported_scan_cpu_seconds
        self._reported_refinement_child_cpu_seconds = (
            reported_refinement_child_cpu_seconds
        )
        self._media_probe = media_probe
        self._embedded_subtitles = embedded_subtitles
        self._pcm_audio_chunks = pcm_audio_chunks
        self._audio_error = audio_error
        self._zero_valid_frames = zero_valid_frames
        self._candidate_proxy_write_count = 0
        self.subtitle_calls: list[tuple[Path, int]] = []
        self.audio_calls: list[tuple[Path, int, int, int]] = []
        self.audio_chunk_calls: list[tuple[Path, int, int, int, int]] = []
        self.extracted_frame_calls: list[tuple[Path, int, int, int]] = []
        self.extracted_original_frame_calls: list[tuple[Path, int, int]] = []
        self.cancel_video_scans_call_count = 0

    def preflight(self) -> MediaRuntimeIdentity:
        """固定runtime identityを返す。"""
        if self._on_preflight is not None:
            self._on_preflight()
        return self._runtime_identity

    def probe(self, media_path: Path) -> MediaProbe:
        """一つのdefault video streamを返す。"""
        self.call_order.append(("probe", media_path.name))
        return self._media_probe or MediaProbe(
            format_names=("matroska",),
            streams=(
                MediaStream(
                    index=0,
                    kind="video",
                    codec_name="ffv1",
                    time_base=Fraction(1, 10),
                    start_pts=0,
                    duration_ts=500 if self._distant_moments else 20,
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

    def scan_pcm_audio(
        self,
        media_path: Path,
        stream_index: int,
        sample_rate: int,
        frame_sample_count: int,
    ) -> Iterator[PcmAudioChunk]:
        """固定PCM chunk列を返す。"""
        self.audio_calls.append(
            (media_path, stream_index, sample_rate, frame_sample_count)
        )
        if self._audio_error is not None:
            raise self._audio_error
        yield from self._pcm_audio_chunks

    def extract_pcm_audio_chunk(
        self,
        media_path: Path,
        stream: MediaStream,
        media_origin: Fraction,
        sample_rate: int,
        sample_start: int,
        maximum_sample_count: int,
    ) -> PcmAudioChunk | None:
        """指定sample rangeに対応する固定PCM chunkを返す。"""
        del media_origin
        self.audio_chunk_calls.append(
            (
                media_path,
                stream.index,
                sample_rate,
                sample_start,
                maximum_sample_count,
            )
        )
        if self._audio_error is not None:
            raise self._audio_error
        return next(
            (
                chunk
                for chunk in self._pcm_audio_chunks
                if chunk.stream_index == stream.index
                and chunk.sample_rate == sample_rate
                and chunk.sample_start >= sample_start
                and chunk.sample_count <= maximum_sample_count
            ),
            None,
        )

    def read_embedded_subtitles(
        self,
        media_path: Path,
        stream_index: int,
    ) -> tuple[EmbeddedSubtitle, ...]:
        """固定embedded subtitle列を返す。"""
        self.subtitle_calls.append((media_path, stream_index))
        return self._embedded_subtitles

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
        scan = self._scan_video_range(
            media_path,
            stream,
            artifact_folder,
            start_pts=None,
            end_pts=None,
            heartbeat_interval_seconds=heartbeat_interval_seconds,
            scene_change_threshold=scene_change_threshold,
            scene_min_interval_seconds=scene_min_interval_seconds,
            decode_backend=decode_backend,
        )
        if isinstance(scan, EmptyVideoScanPartition):
            raise AssertionError("全体scanには1件以上のframeが必要です")
        return scan

    def scan_video_partition(
        self,
        media_path: Path,
        stream: MediaStream,
        artifact_folder: Path,
        *,
        media_origin: Fraction,
        start_pts: int,
        end_pts: int | None,
        heartbeat_interval_seconds: float,
        scene_change_threshold: float,
        scene_min_interval_seconds: float,
        decode_backend: str,
    ) -> NativeVideoScan | EmptyVideoScanPartition:
        """指定PTS区間の決定的なscan結果を生成する。"""
        del media_origin
        self.scan_partition_calls.append((media_path, start_pts, end_pts))
        return self._scan_video_range(
            media_path,
            stream,
            artifact_folder,
            start_pts=start_pts,
            end_pts=end_pts,
            heartbeat_interval_seconds=heartbeat_interval_seconds,
            scene_change_threshold=scene_change_threshold,
            scene_min_interval_seconds=scene_min_interval_seconds,
            decode_backend=decode_backend,
        )

    def _scan_video_range(
        self,
        media_path: Path,
        stream: MediaStream,
        artifact_folder: Path,
        *,
        start_pts: int | None,
        end_pts: int | None,
        heartbeat_interval_seconds: float,
        scene_change_threshold: float,
        scene_min_interval_seconds: float,
        decode_backend: str,
    ) -> NativeVideoScan | EmptyVideoScanPartition:
        """全体または指定PTS区間のfake scanを実行する。"""
        del (
            heartbeat_interval_seconds,
            scene_change_threshold,
            scene_min_interval_seconds,
            decode_backend,
        )
        self.scan_calls.append(media_path)
        self.call_order.append(("scan", media_path.name))
        if self._on_scan_video is not None:
            self._on_scan_video(media_path)
        self._burn_cpu()
        heartbeat_folder = artifact_folder / "heartbeats"
        scene_folder = artifact_folder / ".scene-proxies"
        heartbeat_folder.mkdir(parents=True)
        scene_folder.mkdir()
        range_start = 0 if start_pts is None else start_pts
        available_heartbeat_pts: tuple[int, ...]
        if self._scan_frame_pts is None:
            available_heartbeat_pts = (0, 400) if self._distant_moments else (0, 10)
            available_last_frame_pts = 490 if self._distant_moments else 10
            has_owned_frame = range_start <= available_last_frame_pts
            origin_pts = range_start
        else:
            available_heartbeat_pts = tuple(
                pts
                for pts in self._scan_frame_pts
                if pts >= range_start and (end_pts is None or pts < end_pts)
            )
            has_owned_frame = bool(available_heartbeat_pts)
            available_last_frame_pts = (
                available_heartbeat_pts[-1] if available_heartbeat_pts else range_start
            )
            origin_pts = (
                available_heartbeat_pts[0] if available_heartbeat_pts else range_start
            )
        if not has_owned_frame:
            return EmptyVideoScanPartition(
                stream_index=stream.index,
                start_pts=range_start,
                end_pts=end_pts,
                time_base=Fraction(1, 10),
                wall_seconds=self._reported_scan_wall_seconds,
                cpu_seconds=self._reported_scan_cpu_seconds,
                decode_pass_count=1,
            )
        heartbeat_pts: tuple[int, ...] = available_heartbeat_pts
        if self._scan_frame_pts is None:
            heartbeat_pts = tuple(
                pts
                for pts in available_heartbeat_pts
                if pts >= range_start and (end_pts is None or pts < end_pts)
            )
            last_frame_pts = (
                available_last_frame_pts
                if end_pts is None
                else min(available_last_frame_pts, end_pts - 1)
            )
        else:
            last_frame_pts = available_last_frame_pts
        last_frame_duration_ts = 10
        heartbeats = tuple(
            self._write_scan_frame(heartbeat_folder / f"{pts:012d}.jpg", pts)
            for pts in heartbeat_pts
        )
        available_scene_pts = available_heartbeat_pts[-1]
        scene_pts = (
            (available_scene_pts,)
            if available_scene_pts >= range_start
            and (end_pts is None or available_scene_pts < end_pts)
            else ()
        )
        scenes = tuple(
            self._write_scan_frame(scene_folder / f"{pts:012d}.jpg", pts)
            for pts in scene_pts
        )
        return NativeVideoScan(
            stream_index=stream.index,
            origin_pts=origin_pts,
            last_frame_pts=last_frame_pts,
            last_frame_duration_ts=last_frame_duration_ts,
            time_base=Fraction(1, 10),
            heartbeats=heartbeats,
            scene_frames=scenes,
            wall_seconds=self._reported_scan_wall_seconds,
            cpu_seconds=self._reported_scan_cpu_seconds,
            decode_pass_count=1,
        )

    def cancel_video_scans(self) -> None:
        """scan cancellation要求を記録する。"""
        self.cancel_video_scans_call_count += 1
        if self._on_cancel_video_scans is not None:
            self._on_cancel_video_scans()

    def scan_video_frame_ranges(
        self,
        media_path: Path,
        stream_index: int,
        pts_ranges: tuple[tuple[int, int], ...],
        max_dimension: int,
        *,
        cpu_seconds_recorder: Callable[[float], None] | None = None,
    ) -> Iterator[DecodedVideoFrame]:
        """range内のnative test frameを返す。"""
        del max_dimension
        self.range_calls.append(media_path)
        self.range_pts_calls.append(pts_ranges)
        self.call_order.append(("refine", media_path.name))
        if self._on_scan_video_frame_ranges is not None:
            self._on_scan_video_frame_ranges(media_path)
        self._burn_cpu()
        if cpu_seconds_recorder is not None:
            cpu_seconds_recorder(self._reported_refinement_child_cpu_seconds)
        frame_pts = (0, 5, 395, 400, 405) if self._distant_moments else (0, 5, 10, 15)
        for pts in frame_pts:
            if any(start <= pts < end for start, end in pts_ranges):
                if (
                    self._require_streaming_refinement
                    and pts == 400
                    and self._candidate_proxy_write_count == 0
                ):
                    msg = "次のrefinement groupより前にproxyが書かれていません"
                    raise AssertionError(msg)
                yield self._decoded_frame(stream_index, pts)

    def extract_video_frame(
        self,
        media_path: Path,
        stream_index: int,
        pts: int,
        max_dimension: int,
    ) -> DecodedVideoFrame:
        """指定PTSの元解像度test frameを返す。"""
        self.extracted_frame_calls.append(
            (media_path, stream_index, pts, max_dimension)
        )
        return self._decoded_frame(stream_index, pts)

    def extract_original_video_frame(
        self,
        media_path: Path,
        stream_index: int,
        pts: int,
    ) -> DecodedVideoFrame:
        """指定PTSの元寸法test frameを返す。"""
        self.extracted_original_frame_calls.append((media_path, stream_index, pts))
        return self._decoded_frame(stream_index, pts)

    def write_mjpeg_proxy(
        self,
        frame: DecodedVideoFrame,
        output_path: Path,
        *,
        quality: int,
    ) -> float:
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
        if "candidates" in output_path.parts:
            self._candidate_proxy_write_count += 1
        return 0.0

    def _burn_cpu(self) -> None:
        """Stage resource metric test用にcurrent processのCPUを消費する。"""
        started_at = time.process_time()
        while time.process_time() - started_at < self._cpu_burn_seconds:
            pass

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

    def _decoded_frame(self, stream_index: int, pts: int) -> DecodedVideoFrame:
        if self._zero_valid_frames:
            return DecodedVideoFrame(
                stream_index=stream_index,
                pts=pts,
                duration_ts=5,
                time_base=Fraction(1, 10),
                width=64,
                height=48,
                pixel_format="rgb24",
                pixels=bytes(64 * 48 * 3),
            )
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
