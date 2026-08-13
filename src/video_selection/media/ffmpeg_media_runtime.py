"""system FFmpeg/ffprobeをsemantic media operationへ閉じ込めるruntime。"""

import hashlib
import json
import os
import queue
import re
import signal
import subprocess
import time
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from threading import Lock, Thread
from typing import IO, NoReturn

from ..models.decoded_video_frame import DecodedVideoFrame
from ..models.embedded_subtitle import EmbeddedSubtitle
from ..models.empty_video_scan_partition import EmptyVideoScanPartition
from ..models.media_probe import MediaProbe
from ..models.media_runtime_error import MediaRuntimeError
from ..models.media_runtime_failure_reason import MediaRuntimeFailureReason
from ..models.media_runtime_identity import MediaRuntimeIdentity
from ..models.media_stream import MediaStream
from ..models.native_video_scan import NativeVideoScan
from ..models.pcm_audio_chunk import PcmAudioChunk
from ..models.scanned_video_frame import ScannedVideoFrame
from ..services.read_process_logical_cpu_count import (
    read_process_logical_cpu_count,
)
from ..services.resolve_frame_range_worker_count import (
    resolve_frame_range_worker_count,
)
from ..services.select_scene_signal_frames import select_scene_signal_frames
from .ffmpeg_pcm_reader import iter_pcm_audio_chunks
from .ffmpeg_subtitle_reader import read_embedded_subtitle_events
from .ffmpeg_video_reader import iter_decoded_video_frames
from .ffprobe_parser import parse_media_probe
from .wait_for_process import wait_for_process

_VERSION_PATTERN = re.compile(r"^(?:ffmpeg|ffprobe) version (?P<version>\S+)")
_SEMANTIC_VERSION_PATTERN = re.compile(
    r"(?P<major>\d+)\.(?P<minor>\d+)(?:\.(?P<patch>\d+))?"
)
_CAPABILITY_FLAGS_PATTERN = re.compile(r"[A-Z.]+")
_MINIMUM_VERSION = (6, 1, 1)
_BUILD_SIGNATURE_PREFIXES = (
    "built with ",
    "configuration:",
    "libav",
    "libpostproc",
    "libsw",
)
_REQUIRED_DEMUXERS = frozenset({"matroska", "mov"})
_REQUIRED_DECODERS = frozenset({"aac", "libdav1d", "subrip"})
_REQUIRED_ENCODERS = frozenset({"mjpeg", "pcm_s16le", "ppm", "srt"})
_REQUIRED_MUXERS = frozenset({"image2", "image2pipe", "s16le", "srt"})
_REQUIRED_FILTERS = frozenset(
    {
        "aformat",
        "aresample",
        "asetnsamples",
        "asetpts",
        "asettb",
        "ashowinfo",
        "atrim",
        "concat",
        "format",
        "nullsink",
        "scale",
        "select",
        "setpts",
        "showinfo",
        "split",
    }
)
_DECODE_ERRORS = (
    OSError,
    subprocess.CalledProcessError,
    EOFError,
    ValueError,
)
_SHOWINFO_BRANCH_PATTERN = re.compile(
    r"showinfo@(?P<branch>scan_progress|timeline|heartbeat|scene)"
)
_SHOWINFO_PTS_PATTERN = re.compile(r"\bpts:\s*(-?\d+)")
_SHOWINFO_DURATION_PATTERN = re.compile(r"\bduration:\s*(-?\d+)")
_SHOWINFO_SIZE_PATTERN = re.compile(r"\bs:(\d+)x(\d+)")
_FRAME_RANGE_SEEK_PADDING = Fraction(1)
_FRAME_RANGE_END_PADDING = Fraction(1, 10)
_PCM_PTS_QUANTIZATION_TOLERANCE_SAMPLES = 3
_VIDEO_SCAN_PROGRESS_TIMEOUT_SECONDS = 30 * 60.0
_VIDEO_SCAN_TERMINATION_GRACE_SECONDS = 5.0


class FfmpegMediaRuntime:
    """PATH上のsystem FFmpegとffprobeを使うmedia runtime。"""

    def __init__(
        self,
        ffmpeg_executable: str = "ffmpeg",
        ffprobe_executable: str = "ffprobe",
    ) -> None:
        self._ffmpeg_executable = ffmpeg_executable
        self._ffprobe_executable = ffprobe_executable
        self._active_scan_lock = Lock()
        self._active_scan_processes: set[subprocess.Popen[str]] = set()
        self._video_scan_cancellation_requested = False
        self._active_refinement_lock = Lock()
        self._active_refinement_processes: set[subprocess.Popen[bytes]] = set()
        self._frame_refinement_cancellation_requested = False

    def preflight(self) -> MediaRuntimeIdentity:
        """両toolのversionを解決してidentityを返す。"""
        ffmpeg_version, ffmpeg_build = self._read_identity(
            self._ffmpeg_executable,
            MediaRuntimeFailureReason.FFMPEG_NOT_FOUND,
        )
        ffprobe_version, ffprobe_build = self._read_identity(
            self._ffprobe_executable,
            MediaRuntimeFailureReason.FFPROBE_NOT_FOUND,
        )
        if (
            self._semantic_version(ffmpeg_version) < _MINIMUM_VERSION
            or self._semantic_version(ffprobe_version) < _MINIMUM_VERSION
        ):
            raise MediaRuntimeError(
                MediaRuntimeFailureReason.UNSUPPORTED_FFMPEG_VERSION,
                "FFmpeg/ffprobe 6.1.1以上が必要です",
            )
        if ffmpeg_version != ffprobe_version or ffmpeg_build != ffprobe_build:
            raise MediaRuntimeError(
                MediaRuntimeFailureReason.FFMPEG_FFPROBE_VERSION_MISMATCH,
                "FFmpegとffprobeは同一buildである必要があります",
            )
        capabilities = self._verify_capabilities()
        return MediaRuntimeIdentity(
            ffmpeg_version=ffmpeg_version,
            ffprobe_version=ffprobe_version,
            build_capability_sha256=self._build_capability_sha256(
                ffmpeg_version,
                ffprobe_version,
                ffmpeg_build,
                ffprobe_build,
                capabilities,
            ),
        )

    def probe(self, media_path: Path) -> MediaProbe:
        """containerとordered stream metadataを返す。"""
        try:
            completed = subprocess.run(
                [
                    self._ffprobe_executable,
                    "-v",
                    "error",
                    "-show_error",
                    "-show_format",
                    "-show_streams",
                    "-of",
                    "json",
                    str(media_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            return parse_media_probe(json.loads(completed.stdout))
        except (
            OSError,
            subprocess.CalledProcessError,
            json.JSONDecodeError,
            ValueError,
        ) as error:
            raise MediaRuntimeError(
                MediaRuntimeFailureReason.MEDIA_PROBE_FAILED,
                "media containerをprobeできませんでした",
            ) from error

    def scan_video_frames(
        self,
        media_path: Path,
        stream_index: int,
        max_dimension: int,
    ) -> Iterator[DecodedVideoFrame]:
        """一回のdecodeからnative PTS順にRGB24 proxy frameを返す。"""
        if max_dimension < 1:
            msg = "max_dimensionは正の整数である必要があります"
            raise ValueError(msg)
        command = self._video_decode_command(
            media_path,
            stream_index,
            _scale_filter(max_dimension),
        )
        try:
            yield from iter_decoded_video_frames(command, stream_index)
        except _DECODE_ERRORS as error:
            raise MediaRuntimeError(
                MediaRuntimeFailureReason.DECODER_FAILURE,
                "video streamをdecodeできませんでした",
            ) from error

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
        """一回のdecodeをheartbeat、scene、timeline timingへ分岐する。"""
        scan = self._execute_video_scan(
            media_path,
            stream,
            artifact_folder,
            media_origin=None,
            start_pts=None,
            end_pts=None,
            heartbeat_interval_seconds=heartbeat_interval_seconds,
            scene_change_threshold=scene_change_threshold,
            scene_min_interval_seconds=scene_min_interval_seconds,
            decode_backend=decode_backend,
        )
        if isinstance(scan, EmptyVideoScanPartition):
            raise MediaRuntimeError(
                MediaRuntimeFailureReason.DECODER_FAILURE,
                "Video Scanに表示可能frameがありません",
            )
        selected_scenes = select_scene_signal_frames(
            scan.scene_frames,
            scene_min_interval_seconds,
        )
        if not scan.heartbeats:
            raise MediaRuntimeError(
                MediaRuntimeFailureReason.DECODER_FAILURE,
                "Video Scanにheartbeat frameがありません",
            )
        selected_paths = {frame.image_path for frame in selected_scenes}
        for frame in scan.scene_frames:
            if frame.image_path not in selected_paths:
                frame.image_path.unlink()
        return replace(scan, scene_frames=selected_scenes)

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
        """固定半開PTS区間または指定PTSからEOFまでを一回decodeする。"""
        if end_pts is not None and start_pts >= end_pts:
            msg = "Video Scan partitionのPTS rangeが不正です"
            raise ValueError(msg)
        return self._execute_video_scan(
            media_path,
            stream,
            artifact_folder,
            media_origin=media_origin,
            start_pts=start_pts,
            end_pts=end_pts,
            heartbeat_interval_seconds=heartbeat_interval_seconds,
            scene_change_threshold=scene_change_threshold,
            scene_min_interval_seconds=scene_min_interval_seconds,
            decode_backend=decode_backend,
        )

    def _execute_video_scan(
        self,
        media_path: Path,
        stream: MediaStream,
        artifact_folder: Path,
        *,
        media_origin: Fraction | None,
        start_pts: int | None,
        end_pts: int | None,
        heartbeat_interval_seconds: float,
        scene_change_threshold: float,
        scene_min_interval_seconds: float,
        decode_backend: str,
    ) -> NativeVideoScan | EmptyVideoScanPartition:
        """同じscan algorithmを全体または固定partitionへ適用する。"""
        _validate_scan_configuration(
            heartbeat_interval_seconds,
            scene_change_threshold,
            scene_min_interval_seconds,
            decode_backend,
        )
        if (
            stream.kind != "video"
            or stream.time_base is None
            or stream.start_pts is None
        ):
            msg = "Video Scanにはexact start PTSとtime baseが必要です"
            raise ValueError(msg)
        heartbeat_folder = artifact_folder / "heartbeats"
        scene_folder = artifact_folder / ".scene-proxies"
        heartbeat_folder.mkdir(parents=True)
        scene_folder.mkdir()
        command = self._composite_scan_command(
            media_path,
            stream,
            heartbeat_folder,
            scene_folder,
            heartbeat_interval_seconds,
            scene_change_threshold,
            scene_min_interval_seconds,
            decode_backend,
            media_origin=media_origin,
            start_pts=start_pts,
            end_pts=end_pts,
        )
        started_at = time.monotonic()
        timeline_first: tuple[int, int | None, int, int] | None = None
        timeline_last: tuple[int, int | None, int, int] | None = None
        previous_timeline_pts: int | None = None
        minimum_frame_delta_ts: int | None = None
        current_frame_count_per_pts = 0
        maximum_frame_count_per_pts = 0
        maximum_frame_width = 0
        maximum_frame_height = 0
        frame_timing_reliable = True
        heartbeat_metadata: list[tuple[int, int | None, int, int]] = []
        scene_metadata: list[tuple[int, int | None, int, int]] = []
        stderr_tail: deque[str] = deque(maxlen=80)
        process: subprocess.Popen[str] | None = None
        stderr_thread: Thread | None = None
        try:
            with self._active_scan_lock:
                if self._video_scan_cancellation_requested:
                    raise MediaRuntimeError(
                        MediaRuntimeFailureReason.DECODER_FAILURE,
                        "Video Scanは開始前にcancelされました",
                    )
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                self._active_scan_processes.add(process)
            if process.stderr is None:
                msg = "FFmpeg scan stderrを開始できません"
                raise RuntimeError(msg)
            stderr_queue: queue.Queue[str | BaseException | None] = queue.Queue()
            stderr_thread = Thread(
                target=_collect_scan_stderr,
                args=(process.stderr, stderr_queue),
                daemon=True,
            )
            stderr_thread.start()
            progress_deadline = time.monotonic() + _VIDEO_SCAN_PROGRESS_TIMEOUT_SECONDS
            while True:
                remaining_seconds = progress_deadline - time.monotonic()
                if remaining_seconds <= 0:
                    raise MediaRuntimeError(
                        MediaRuntimeFailureReason.DECODER_STALLED,
                        "Video ScanのFFmpeg decode進捗が停止しました",
                    )
                try:
                    item = stderr_queue.get(timeout=remaining_seconds)
                except queue.Empty as error:
                    raise MediaRuntimeError(
                        MediaRuntimeFailureReason.DECODER_STALLED,
                        "Video ScanのFFmpeg decode進捗が停止しました",
                    ) from error
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                line = item
                stderr_tail.append(line.rstrip())
                parsed = _parse_named_showinfo(line)
                if parsed is None:
                    continue
                branch, metadata = parsed
                if branch == "scan_progress":
                    progress_deadline = (
                        time.monotonic() + _VIDEO_SCAN_PROGRESS_TIMEOUT_SECONDS
                    )
                    continue
                if branch == "timeline":
                    frame_pts, _duration, frame_width, frame_height = metadata
                    maximum_frame_width = max(maximum_frame_width, frame_width)
                    maximum_frame_height = max(maximum_frame_height, frame_height)
                    if previous_timeline_pts is None:
                        current_frame_count_per_pts = 1
                    elif frame_pts == previous_timeline_pts:
                        current_frame_count_per_pts += 1
                    elif frame_pts > previous_timeline_pts:
                        frame_delta_ts = frame_pts - previous_timeline_pts
                        minimum_frame_delta_ts = (
                            frame_delta_ts
                            if minimum_frame_delta_ts is None
                            else min(minimum_frame_delta_ts, frame_delta_ts)
                        )
                        maximum_frame_count_per_pts = max(
                            maximum_frame_count_per_pts,
                            current_frame_count_per_pts,
                        )
                        current_frame_count_per_pts = 1
                    else:
                        frame_timing_reliable = False
                    previous_timeline_pts = frame_pts
                    if timeline_first is None:
                        timeline_first = metadata
                    timeline_last = metadata
                elif branch == "heartbeat":
                    heartbeat_metadata.append(metadata)
                else:
                    scene_metadata.append(metadata)
            remaining_seconds = progress_deadline - time.monotonic()
            if remaining_seconds <= 0:
                raise MediaRuntimeError(
                    MediaRuntimeFailureReason.DECODER_STALLED,
                    "Video ScanのFFmpeg decode進捗が停止しました",
                )
            return_code, cpu_seconds = _wait_for_scan_process_with_timeout(
                process,
                remaining_seconds,
            )
        except OSError as error:
            raise MediaRuntimeError(
                MediaRuntimeFailureReason.DECODER_FAILURE,
                "Video ScanのFFmpeg processを開始できませんでした",
            ) from error
        finally:
            if process is not None:
                if process.returncode is None:
                    _terminate_and_reap_scan_process(process)
                if process.stderr is not None:
                    with suppress(OSError):
                        process.stderr.close()
                if stderr_thread is not None:
                    stderr_thread.join()
                with self._active_scan_lock:
                    self._active_scan_processes.discard(process)
        wall_seconds = time.monotonic() - started_at
        if return_code != 0:
            detail = "\n".join(stderr_tail)
            raise MediaRuntimeError(
                MediaRuntimeFailureReason.DECODER_FAILURE,
                f"Video ScanのFFmpeg decodeに失敗しました\n{detail}",
            )
        if timeline_first is None or timeline_last is None:
            if start_pts is not None:
                if any(heartbeat_folder.iterdir()) or any(scene_folder.iterdir()):
                    raise MediaRuntimeError(
                        MediaRuntimeFailureReason.DECODER_FAILURE,
                        "Video Scanのproxyに対応するtimingを解析できませんでした",
                    )
                return EmptyVideoScanPartition(
                    stream_index=stream.index,
                    start_pts=start_pts,
                    end_pts=end_pts,
                    time_base=stream.time_base,
                    wall_seconds=wall_seconds,
                    cpu_seconds=cpu_seconds,
                    decode_pass_count=1,
                )
            raise MediaRuntimeError(
                MediaRuntimeFailureReason.DECODER_FAILURE,
                "Video Scanに表示可能frameがありません",
            )
        heartbeat_files = _remove_scan_proxy_sentinel(heartbeat_folder)
        scene_files = _remove_scan_proxy_sentinel(scene_folder)
        if len(heartbeat_files) != len(heartbeat_metadata) or len(scene_files) != len(
            scene_metadata
        ):
            raise MediaRuntimeError(
                MediaRuntimeFailureReason.DECODER_FAILURE,
                "Video Scanのproxyとexact timingの件数が一致しません",
            )
        heartbeats = _scanned_frames(
            heartbeat_metadata,
            heartbeat_files,
            stream.time_base,
        )
        scene_frames = tuple(
            _scanned_frames(
                scene_metadata,
                scene_files,
                stream.time_base,
            )
        )
        origin_pts, _origin_duration, _origin_width, _origin_height = timeline_first
        last_pts, last_duration, _last_width, _last_height = timeline_last
        maximum_frame_count_per_pts = max(
            maximum_frame_count_per_pts,
            current_frame_count_per_pts,
        )
        if not frame_timing_reliable or minimum_frame_delta_ts is None:
            minimum_frame_delta_ts = None
            resolved_maximum_frame_count_per_pts = None
        else:
            resolved_maximum_frame_count_per_pts = maximum_frame_count_per_pts
        return NativeVideoScan(
            stream_index=stream.index,
            origin_pts=origin_pts,
            last_frame_pts=last_pts,
            last_frame_duration_ts=last_duration,
            time_base=stream.time_base,
            heartbeats=tuple(heartbeats),
            scene_frames=scene_frames,
            wall_seconds=wall_seconds,
            cpu_seconds=cpu_seconds,
            decode_pass_count=1,
            minimum_frame_delta_ts=minimum_frame_delta_ts,
            maximum_frame_count_per_pts=resolved_maximum_frame_count_per_pts,
            maximum_frame_width=maximum_frame_width,
            maximum_frame_height=maximum_frame_height,
        )

    def cancel_video_scans(self) -> None:
        """実行中のVideo Scan FFmpeg processへ終了要求を送る。"""
        with self._active_scan_lock:
            self._video_scan_cancellation_requested = True
            processes = tuple(self._active_scan_processes)
        for process in processes:
            with suppress(OSError):
                os.kill(process.pid, signal.SIGTERM)

    def cancel_frame_refinements(self) -> None:
        """実行中のFrame Refinement FFmpeg processへ終了要求を送る。"""
        with self._active_refinement_lock:
            self._frame_refinement_cancellation_requested = True
            processes = tuple(self._active_refinement_processes)
        for process in processes:
            with suppress(OSError):
                os.kill(process.pid, signal.SIGTERM)

    def scan_video_frame_ranges(
        self,
        media_path: Path,
        stream_index: int,
        pts_ranges: tuple[tuple[int, int], ...],
        max_dimension: int,
        *,
        cpu_seconds_recorder: Callable[[float], None] | None = None,
    ) -> Iterator[DecodedVideoFrame]:
        """半開PTS rangeの和集合にあるnative RGB24 frameだけを返す。"""
        if (
            max_dimension < 1
            or not pts_ranges
            or any(start >= end for start, end in pts_ranges)
        ):
            msg = "PTS rangeとmax_dimensionが不正です"
            raise ValueError(msg)
        with self._active_refinement_lock:
            if self._frame_refinement_cancellation_requested:
                raise MediaRuntimeError(
                    MediaRuntimeFailureReason.FRAME_EXTRACTION_FAILED,
                    "Frame Refinementは開始前にcancelされました",
                )
        probe = self.probe(media_path)
        try:
            stream = next(
                item
                for item in probe.streams
                if item.index == stream_index and item.kind == "video"
            )
            if stream.time_base is None or stream.start_pts is None:
                msg = "Frame Refinementには開始PTSを持つvideo streamが必要です"
                raise ValueError(msg)
            media_origin = _media_origin(probe)
            commands = tuple(
                self._video_range_decode_command(
                    media_path,
                    stream,
                    start,
                    end,
                    media_origin,
                    max_dimension,
                )
                for start, end in pts_ranges
            )
            worker_count = resolve_frame_range_worker_count(
                len(commands),
                logical_cpu_count=read_process_logical_cpu_count(),
            )
            with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="frame-range-decode",
            ) as executor:
                command_iterator = iter(commands)
                pending: deque[Future[tuple[DecodedVideoFrame, ...]]] = deque()
                for _ in range(worker_count):
                    pending.append(
                        executor.submit(
                            _decode_video_frame_range,
                            next(command_iterator),
                            stream_index,
                            cpu_seconds_recorder,
                            self._register_refinement_process,
                            self._unregister_refinement_process,
                        )
                    )
                while pending:
                    yield from pending.popleft().result()
                    command = next(command_iterator, None)
                    if command is not None:
                        pending.append(
                            executor.submit(
                                _decode_video_frame_range,
                                command,
                                stream_index,
                                cpu_seconds_recorder,
                                self._register_refinement_process,
                                self._unregister_refinement_process,
                            )
                        )
        except (*_DECODE_ERRORS, StopIteration) as error:
            raise MediaRuntimeError(
                MediaRuntimeFailureReason.FRAME_EXTRACTION_FAILED,
                "指定されたPTS rangeのnative frameを抽出できませんでした",
            ) from error

    def _register_refinement_process(
        self,
        process: subprocess.Popen[bytes],
    ) -> None:
        """active decoderを登録し先行cancel済みなら直ちに終了させる。"""
        with self._active_refinement_lock:
            self._active_refinement_processes.add(process)
            cancellation_requested = self._frame_refinement_cancellation_requested
        if cancellation_requested:
            with suppress(OSError):
                os.kill(process.pid, signal.SIGTERM)

    def _unregister_refinement_process(
        self,
        process: subprocess.Popen[bytes],
    ) -> None:
        """終了済みdecoderをactive集合から解除する。"""
        with self._active_refinement_lock:
            self._active_refinement_processes.discard(process)

    def extract_video_frame(
        self,
        media_path: Path,
        stream_index: int,
        pts: int,
        max_dimension: int,
    ) -> DecodedVideoFrame:
        """指定source PTSの一つのRGB24 frameを返す。"""
        if max_dimension < 1:
            msg = "max_dimensionは正の整数である必要があります"
            raise ValueError(msg)
        frame_filter = f"select=eq(pts\\,{pts})," + _scale_filter(max_dimension)
        return self._extract_exact_video_frame(
            media_path,
            stream_index,
            pts,
            frame_filter,
        )

    def extract_original_video_frame(
        self,
        media_path: Path,
        stream_index: int,
        pts: int,
    ) -> DecodedVideoFrame:
        """指定source PTSの一つの元寸法RGB24 frameを返す。"""
        return self._extract_exact_video_frame(
            media_path,
            stream_index,
            pts,
            f"select=eq(pts\\,{pts}),format=rgb24,showinfo",
        )

    def _extract_exact_video_frame(
        self,
        media_path: Path,
        stream_index: int,
        pts: int,
        frame_filter: str,
    ) -> DecodedVideoFrame:
        """入力seek後に指定filterでexact PTSのRGB24 frameを一つだけ返す。"""
        try:
            probe = self.probe(media_path)
            stream = next(
                item
                for item in probe.streams
                if item.index == stream_index and item.kind == "video"
            )
            if stream.time_base is None or stream.start_pts is None:
                msg = "Exact Frame Extractionには開始PTSが必要です"
                raise ValueError(msg)
            input_options = self._exact_frame_input_options(
                stream,
                pts,
                _media_origin(probe),
            )
            command = self._video_decode_command(
                media_path,
                stream_index,
                frame_filter,
                frame_limit=1,
                input_options=input_options,
            )
            frames = tuple(iter_decoded_video_frames(command, stream_index))
        except (*_DECODE_ERRORS, StopIteration) as error:
            raise MediaRuntimeError(
                MediaRuntimeFailureReason.FRAME_EXTRACTION_FAILED,
                "指定されたvideo frameを抽出できませんでした",
            ) from error
        if len(frames) != 1 or frames[0].pts != pts:
            raise MediaRuntimeError(
                MediaRuntimeFailureReason.FRAME_EXTRACTION_FAILED,
                "指定されたsource PTSのvideo frameがありません",
            )
        return frames[0]

    @staticmethod
    def _exact_frame_input_options(
        stream: MediaStream,
        pts: int,
        media_origin: Fraction,
    ) -> tuple[str, ...]:
        """Exact PTSの直前からdecodeするinput seek範囲を返す。"""
        if stream.time_base is None:
            msg = "Exact Frame Extraction streamのtime baseがありません"
            raise ValueError(msg)
        relative_target = pts * stream.time_base - media_origin
        if relative_target < 0:
            msg = "Exact Frame Extraction PTSがmedia originより前です"
            raise ValueError(msg)
        seek_padding = min(_FRAME_RANGE_SEEK_PADDING, relative_target)
        seek_seconds = relative_target - seek_padding
        read_seconds = seek_padding + _FRAME_RANGE_END_PADDING
        return (
            "-ss",
            _ffmpeg_number(float(seek_seconds)),
            "-t",
            _ffmpeg_number(float(read_seconds)),
        )

    def write_mjpeg_proxy(
        self,
        frame: DecodedVideoFrame,
        output_path: Path,
        *,
        quality: int,
    ) -> float:
        """RGB24 artifactをFFmpeg MJPEGへmetadataなしで保存する。"""
        if not 1 <= quality <= 31:
            msg = "FFmpeg MJPEG qualityは1以上31以下である必要があります"
            raise ValueError(msg)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            self._ffmpeg_executable,
            "-y",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            frame.pixel_format,
            "-video_size",
            f"{frame.width}x{frame.height}",
            "-framerate",
            "1",
            "-i",
            "pipe:0",
            "-frames:v",
            "1",
            "-map_metadata",
            "-1",
            "-c:v",
            "mjpeg",
            "-q:v",
            str(quality),
            "-pix_fmt",
            "yuvj420p",
            str(output_path),
        ]
        process: subprocess.Popen[bytes] | None = None
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            if process.stdin is None or process.stderr is None:
                raise RuntimeError("Frame Candidate Proxy pipeを開始できません")
            process.stdin.write(frame.pixels)
            process.stdin.close()
            stderr = process.stderr.read()
            return_code, cpu_seconds = wait_for_process(process)
            if return_code != 0:
                raise subprocess.CalledProcessError(
                    return_code,
                    command,
                    stderr=stderr,
                )
            return cpu_seconds
        except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
            raise MediaRuntimeError(
                MediaRuntimeFailureReason.FRAME_EXTRACTION_FAILED,
                "Frame Candidate ProxyをMJPEGへencodeできませんでした",
            ) from error
        finally:
            if process is not None:
                if process.returncode is None:
                    with suppress(ProcessLookupError):
                        os.kill(process.pid, signal.SIGTERM)
                    with suppress(ChildProcessError):
                        wait_for_process(process)
                if process.stdin is not None:
                    process.stdin.close()
                if process.stderr is not None:
                    process.stderr.close()

    def scan_pcm_audio(
        self,
        media_path: Path,
        stream_index: int,
        sample_rate: int,
        frame_sample_count: int,
    ) -> Iterator[PcmAudioChunk]:
        """選択audioをmono s16leの連続sample gridとして返す。"""
        if sample_rate < 1 or frame_sample_count < 1:
            msg = "sample_rateとframe_sample_countは正の整数が必要です"
            raise ValueError(msg)
        audio_filter = (
            f"aresample={sample_rate}:async=0,"
            "aformat=sample_fmts=s16:channel_layouts=mono,"
            f"asetnsamples=n={frame_sample_count}:p=0,"
            "asetpts=N/SR/TB+STARTPTS,ashowinfo"
        )
        command = self._decode_command_prefix(media_path, stream_index)
        command.extend(
            [
                "-vn",
                "-sn",
                "-dn",
                "-af",
                audio_filter,
                "-f",
                "s16le",
                "-acodec",
                "pcm_s16le",
                "pipe:1",
            ]
        )
        try:
            yield from iter_pcm_audio_chunks(
                command,
                stream_index,
                sample_rate,
            )
        except _DECODE_ERRORS as error:
            raise MediaRuntimeError(
                MediaRuntimeFailureReason.AUDIO_EXTRACTION_FAILED,
                "audio streamをPCMへdecodeできませんでした",
            ) from error

    def extract_pcm_audio_chunk(
        self,
        media_path: Path,
        stream: MediaStream,
        media_origin: Fraction,
        sample_rate: int,
        sample_start: int,
        maximum_sample_count: int,
    ) -> PcmAudioChunk | None:
        """一つのcanonical sample rangeだけを入力seek付きで抽出する。"""
        if (
            stream.kind != "audio"
            or stream.start_pts is None
            or stream.time_base is None
            or sample_rate < 1
            or sample_start < 0
            or maximum_sample_count < 1
        ):
            msg = "PCM Audio rangeのstreamまたはsample設定が不正です"
            raise ValueError(msg)
        stream_origin = stream.start_pts * stream.time_base
        relative_start = (
            stream_origin - media_origin + Fraction(sample_start, sample_rate)
        )
        if relative_start < 0:
            msg = "PCM Audio rangeがmedia originより前です"
            raise ValueError(msg)
        range_duration = _ffmpeg_number(maximum_sample_count / sample_rate)
        input_options = (
            ("-t", range_duration)
            if relative_start == 0
            else (
                "-ss",
                _ffmpeg_number(float(relative_start)),
                "-t",
                range_duration,
            )
        )
        absolute_start_pts = round(stream_origin * sample_rate) + sample_start
        audio_filter = (
            f"aresample={sample_rate}:async=0,"
            "aformat=sample_fmts=s16:channel_layouts=mono,"
            f"atrim=end_sample={maximum_sample_count},"
            f"asettb=expr=1/{sample_rate},"
            "ashowinfo@observed,"
            f"asetnsamples=n={maximum_sample_count}:p=0,"
            "ashowinfo@chunk"
        )
        command = self._decode_command_prefix(
            media_path,
            stream.index,
            input_options=input_options,
        )
        command.extend(
            [
                "-vn",
                "-sn",
                "-dn",
                "-af",
                audio_filter,
                "-f",
                "s16le",
                "-acodec",
                "pcm_s16le",
                "pipe:1",
            ]
        )
        try:
            observed_timing: list[tuple[int, int]] = []
            chunks = tuple(
                iter_pcm_audio_chunks(
                    command,
                    stream.index,
                    sample_rate,
                    initial_sample_start=sample_start,
                    observed_timing=observed_timing,
                )
            )
        except _DECODE_ERRORS as error:
            raise MediaRuntimeError(
                MediaRuntimeFailureReason.AUDIO_EXTRACTION_FAILED,
                "audio sample rangeをPCMへdecodeできませんでした",
            ) from error
        if not chunks:
            return None
        if len(chunks) != 1 or chunks[0].sample_count > maximum_sample_count:
            raise MediaRuntimeError(
                MediaRuntimeFailureReason.AUDIO_EXTRACTION_FAILED,
                "audio sample rangeがcanonical chunkへ収まりません",
            )
        chunk = chunks[0]
        if not _pcm_range_timing_is_continuous(
            observed_timing,
            absolute_start_pts,
            chunk.sample_count,
        ):
            raise MediaRuntimeError(
                MediaRuntimeFailureReason.AUDIO_EXTRACTION_FAILED,
                "audio sample rangeのtimestampが連続していません",
            )
        return replace(chunk, pts=absolute_start_pts)

    def read_embedded_subtitles(
        self,
        media_path: Path,
        stream_index: int,
    ) -> tuple[EmbeddedSubtitle, ...]:
        """選択text subtitleの元packet timingとdecoded textを返す。"""
        try:
            stream = next(
                item
                for item in self.probe(media_path).streams
                if item.index == stream_index and item.kind == "subtitle"
            )
            if stream.time_base is None:
                msg = "subtitle streamに有効なtime baseがありません"
                raise ValueError(msg)
            return read_embedded_subtitle_events(
                self._ffmpeg_executable,
                self._ffprobe_executable,
                media_path,
                stream_index,
                stream.time_base,
            )
        except (
            OSError,
            StopIteration,
            subprocess.CalledProcessError,
            json.JSONDecodeError,
            UnicodeError,
            ValueError,
        ) as error:
            raise MediaRuntimeError(
                MediaRuntimeFailureReason.SUBTITLE_EXTRACTION_FAILED,
                "embedded text subtitleを抽出できませんでした",
            ) from error

    def _video_decode_command(
        self,
        media_path: Path,
        stream_index: int,
        frame_filter: str,
        frame_limit: int | None = None,
        input_options: tuple[str, ...] = (),
    ) -> list[str]:
        command = self._decode_command_prefix(
            media_path,
            stream_index,
            input_options=input_options,
        )
        command.extend(["-an", "-sn", "-dn", "-vf", frame_filter])
        if frame_limit is not None:
            command.extend(["-frames:v", str(frame_limit)])
        command.extend(
            [
                "-fps_mode",
                "passthrough",
                "-f",
                "image2pipe",
                "-c:v",
                "ppm",
                "pipe:1",
            ]
        )
        return command

    def _video_range_decode_command(
        self,
        media_path: Path,
        stream: MediaStream,
        start_pts: int,
        end_pts: int,
        media_origin: Fraction,
        max_dimension: int,
    ) -> list[str]:
        """一つの半開PTS rangeだけを入力seek付きでdecodeする。"""
        if stream.time_base is None:
            msg = "Frame Refinement streamのtime baseがありません"
            raise ValueError(msg)
        relative_start = start_pts * stream.time_base - media_origin
        if relative_start < 0:
            msg = "Frame Refinement rangeがmedia originより前です"
            raise ValueError(msg)
        seek_padding = min(_FRAME_RANGE_SEEK_PADDING, relative_start)
        seek_seconds = relative_start - seek_padding
        read_seconds = (
            seek_padding
            + (end_pts - start_pts) * stream.time_base
            + _FRAME_RANGE_END_PADDING
        )
        frame_filter = (
            f"select='gte(pts\\,{start_pts})*lt(pts\\,{end_pts})',"
            + _scale_filter(max_dimension)
        )
        return self._video_decode_command(
            media_path,
            stream.index,
            frame_filter,
            input_options=(
                "-ss",
                _ffmpeg_number(float(seek_seconds)),
                "-t",
                _ffmpeg_number(float(read_seconds)),
            ),
        )

    def _composite_scan_command(
        self,
        media_path: Path,
        stream: MediaStream,
        heartbeat_folder: Path,
        scene_folder: Path,
        heartbeat_interval_seconds: float,
        scene_change_threshold: float,
        scene_min_interval_seconds: float,
        decode_backend: str,
        *,
        media_origin: Fraction | None,
        start_pts: int | None,
        end_pts: int | None,
    ) -> list[str]:
        if stream.start_pts is None:
            msg = "Video Scanにはstream start PTSが必要です"
            raise ValueError(msg)
        command = [
            self._ffmpeg_executable,
            "-y",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "info",
            "-nostats",
            "-xerror",
            "-err_detect",
            "explode",
            "-copyts",
        ]
        if decode_backend == "nvdec":
            command.extend(["-hwaccel", "cuda"])
        elif stream.codec_name == "av1":
            command.extend(["-c:v", "libdav1d"])
        if start_pts is not None:
            if (
                stream.time_base is None
                or stream.start_pts is None
                or media_origin is None
            ):
                msg = "Video Scan partitionにはstream timingが必要です"
                raise ValueError(msg)
            relative_start = start_pts * stream.time_base - media_origin
            if relative_start < 0:
                msg = "Video Scan partitionがmedia originより前です"
                raise ValueError(msg)
            required_preroll = (
                max(
                    _FRAME_RANGE_SEEK_PADDING,
                    Fraction(str(heartbeat_interval_seconds)),
                    Fraction(str(scene_min_interval_seconds)),
                )
                + _FRAME_RANGE_END_PADDING
            )
            seek_padding = min(required_preroll, relative_start)
            command.extend(
                [
                    "-ss",
                    _ffmpeg_number(float(relative_start - seek_padding)),
                ]
            )
            if end_pts is not None:
                read_seconds = (
                    seek_padding
                    + (end_pts - start_pts) * stream.time_base
                    + _FRAME_RANGE_END_PADDING
                )
                command.extend(["-t", _ffmpeg_number(float(read_seconds))])
        command.extend(["-i", str(media_path)])
        heartbeat_interval = _ffmpeg_number(heartbeat_interval_seconds)
        scene_threshold = _ffmpeg_number(scene_change_threshold)
        partition_filter = _scan_partition_filter(start_pts, end_pts)
        ownership_filter = "" if partition_filter is None else f"{partition_filter},"
        source_graph = (
            f"[0:{stream.index}]showinfo@scan_progress,split=5[timeline_source]"
            "[heartbeat_source][heartbeat_sentinel_source]"
            "[scene_source][scene_sentinel_source]"
        )
        heartbeat_filter = _canonical_heartbeat_filter(
            stream.start_pts,
            heartbeat_interval,
        )
        graph = ";".join(
            (
                source_graph,
                (f"[timeline_source]{ownership_filter}showinfo@timeline,nullsink"),
                (
                    "[heartbeat_source]"
                    f"{heartbeat_filter},"
                    f"{ownership_filter}"
                    f"{_bounded_scale_filter(960)},"
                    "format=yuvj420p,showinfo@heartbeat,"
                    "setpts=PTS-STARTPTS[heartbeat_actual]"
                ),
                (
                    "[heartbeat_sentinel_source]"
                    f"{ownership_filter}"
                    "select='eq(n,0)',"
                    f"{_bounded_scale_filter(960)},"
                    "format=yuvj420p,setpts=PTS-STARTPTS[heartbeat_sentinel]"
                ),
                (
                    "[heartbeat_sentinel][heartbeat_actual]"
                    "concat=n=2:v=1:a=0,setpts=N/(1*TB)[heartbeat_output]"
                ),
                (
                    "[scene_source]"
                    f"{_bounded_scale_filter(320)},"
                    f"select='gt(scene,{scene_threshold})',"
                    f"{ownership_filter}"
                    "format=yuvj420p,showinfo@scene,"
                    "setpts=PTS-STARTPTS[scene_actual]"
                ),
                (
                    "[scene_sentinel_source]"
                    f"{ownership_filter}"
                    "select='eq(n,0)',"
                    f"{_bounded_scale_filter(320)},"
                    "format=yuvj420p,setpts=PTS-STARTPTS[scene_sentinel]"
                ),
                (
                    "[scene_sentinel][scene_actual]"
                    "concat=n=2:v=1:a=0,setpts=N/(1*TB)[scene_output]"
                ),
            )
        )
        command.extend(
            [
                "-filter_complex",
                graph,
                "-map",
                "[heartbeat_output]",
                "-fps_mode",
                "passthrough",
                "-map_metadata",
                "-1",
                "-q:v",
                "3",
                str(heartbeat_folder / "%012d.jpg"),
                "-map",
                "[scene_output]",
                "-fps_mode",
                "passthrough",
                "-map_metadata",
                "-1",
                "-q:v",
                "3",
                str(scene_folder / "%012d.jpg"),
            ]
        )
        return command

    def _decode_command_prefix(
        self,
        media_path: Path,
        stream_index: int,
        *,
        input_options: tuple[str, ...] = (),
    ) -> list[str]:
        return [
            self._ffmpeg_executable,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "info",
            "-nostats",
            "-xerror",
            "-err_detect",
            "explode",
            "-copyts",
            *input_options,
            "-i",
            str(media_path),
            "-map",
            f"0:{stream_index}",
        ]

    @staticmethod
    def _read_identity(
        executable: str,
        missing_reason: MediaRuntimeFailureReason,
    ) -> tuple[str, tuple[str, ...]]:
        try:
            completed = subprocess.run(
                [executable, "-version"],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            msg = f"{missing_reason.value}"
            raise MediaRuntimeError(missing_reason, msg) from error
        output_lines = completed.stdout.splitlines()
        if not output_lines:
            raise MediaRuntimeError(
                MediaRuntimeFailureReason.UNSUPPORTED_FFMPEG_VERSION,
                "FFmpeg toolのversion outputがありません",
            )
        first_line = output_lines[0]
        match = _VERSION_PATTERN.match(first_line)
        if match is None:
            raise MediaRuntimeError(
                MediaRuntimeFailureReason.UNSUPPORTED_FFMPEG_VERSION,
                "FFmpeg toolのversionを解決できません",
            )
        build_signature = tuple(
            line.strip()
            for line in output_lines[1:]
            if line.strip().startswith(_BUILD_SIGNATURE_PREFIXES)
        )
        return match.group("version"), build_signature

    @staticmethod
    def _semantic_version(version: str) -> tuple[int, int, int]:
        match = _SEMANTIC_VERSION_PATTERN.search(version)
        if match is None:
            raise MediaRuntimeError(
                MediaRuntimeFailureReason.UNSUPPORTED_FFMPEG_VERSION,
                "FFmpeg toolのversion形式を解釈できません",
            )
        patch = match.group("patch")
        return (
            int(match.group("major")),
            int(match.group("minor")),
            int(patch) if patch is not None else 0,
        )

    def _verify_capabilities(self) -> dict[str, object]:
        try:
            demuxers = self._read_capability_names("-demuxers")
            decoders = self._read_capability_names("-decoders")
            encoders = self._read_capability_names("-encoders")
            muxers = self._read_capability_names("-muxers")
            filters = self._read_capability_names("-filters")
            probe = subprocess.run(
                [
                    self._ffprobe_executable,
                    "-v",
                    "error",
                    "-show_program_version",
                    "-of",
                    "json",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            probe_document = json.loads(probe.stdout)
        except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
            self._raise_missing_capability(error)
        if not isinstance(probe_document, dict):
            self._raise_missing_capability()
        program_version = probe_document.get("program_version")
        if (
            not _REQUIRED_DEMUXERS.issubset(demuxers)
            or not _REQUIRED_DECODERS.issubset(decoders)
            or not _REQUIRED_ENCODERS.issubset(encoders)
            or not _REQUIRED_MUXERS.issubset(muxers)
            or not _REQUIRED_FILTERS.issubset(filters)
            or not isinstance(program_version, dict)
        ):
            self._raise_missing_capability()
        return {
            "demuxers": sorted(demuxers),
            "decoders": sorted(decoders),
            "encoders": sorted(encoders),
            "muxers": sorted(muxers),
            "filters": sorted(filters),
            "ffprobe_program_version": program_version,
        }

    @staticmethod
    def _build_capability_sha256(
        ffmpeg_version: str,
        ffprobe_version: str,
        ffmpeg_build: tuple[str, ...],
        ffprobe_build: tuple[str, ...],
        capabilities: dict[str, object],
    ) -> str:
        """raw build文字列を残さずcanonical identity digestを導出する。"""
        canonical = json.dumps(
            {
                "ffmpeg_version": ffmpeg_version,
                "ffprobe_version": ffprobe_version,
                "ffmpeg_build": list(ffmpeg_build),
                "ffprobe_build": list(ffprobe_build),
                "capabilities": capabilities,
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def _read_capability_names(self, option: str) -> frozenset[str]:
        completed = subprocess.run(
            [self._ffmpeg_executable, "-hide_banner", option],
            check=True,
            capture_output=True,
            text=True,
        )
        names: set[str] = set()
        for line in completed.stdout.splitlines():
            fields = line.split()
            if (
                len(fields) < 2
                or _CAPABILITY_FLAGS_PATTERN.fullmatch(fields[0]) is None
            ):
                continue
            names.update(fields[1].split(","))
        return frozenset(names)

    @staticmethod
    def _raise_missing_capability(error: BaseException | None = None) -> NoReturn:
        failure = MediaRuntimeError(
            MediaRuntimeFailureReason.MISSING_REQUIRED_DEMUXER_OR_DECODER,
            "必要なFFmpeg/ffprobe media能力がありません",
        )
        if error is None:
            raise failure
        raise failure from error


def _scale_filter(max_dimension: int) -> str:
    return (
        f"scale=w=min(iw\\,{max_dimension}):h=min(ih\\,{max_dimension}):"
        "force_original_aspect_ratio=decrease:force_divisible_by=2,"
        "format=rgb24,showinfo"
    )


def _scan_partition_filter(
    start_pts: int | None,
    end_pts: int | None,
) -> str | None:
    if start_pts is None:
        return None
    if end_pts is None:
        return f"select='gte(pts\\,{start_pts})'"
    return f"select='gte(pts\\,{start_pts})*lt(pts\\,{end_pts})'"


def _canonical_heartbeat_filter(
    origin_pts: int,
    heartbeat_interval: str,
) -> str:
    """global time bucketごとの先頭frameをpartition非依存に選ぶ。"""
    current_bucket = f"floor((pts-{origin_pts})*TB/{heartbeat_interval})"
    previous_bucket = f"floor((prev_pts-{origin_pts})*TB/{heartbeat_interval})"
    return f"select='isnan(prev_pts)+gt({current_bucket},{previous_bucket})'"


def _media_origin(probe: MediaProbe) -> Fraction:
    origins = tuple(
        stream.start_pts * stream.time_base
        for stream in probe.streams
        if stream.start_pts is not None and stream.time_base is not None
    )
    if not origins:
        msg = "media streamに開始PTSがありません"
        raise ValueError(msg)
    return min(origins)


def _decode_video_frame_range(
    command: list[str],
    stream_index: int,
    cpu_seconds_recorder: Callable[[float], None] | None,
    on_process_started: Callable[[subprocess.Popen[bytes]], None],
    on_process_finished: Callable[[subprocess.Popen[bytes]], None],
) -> tuple[DecodedVideoFrame, ...]:
    """一つのrangeをworker内で完結させ順序付きframeを返す。"""
    return tuple(
        iter_decoded_video_frames(
            command,
            stream_index,
            cpu_seconds_recorder=cpu_seconds_recorder,
            on_process_started=on_process_started,
            on_process_finished=on_process_finished,
        )
    )


def _collect_scan_stderr(
    stderr: IO[str],
    stderr_queue: queue.Queue[str | BaseException | None],
) -> None:
    """blocking stderrを別threadで読みmain threadの進捗期限を有効にする。"""
    try:
        for line in stderr:
            stderr_queue.put(line)
    except BaseException as error:
        stderr_queue.put(error)
    finally:
        stderr_queue.put(None)


def _wait_for_scan_process_with_timeout(
    process: subprocess.Popen[str],
    timeout_seconds: float,
) -> tuple[int, float]:
    """一つのreaperでCPU metricとstall期限付き終了を回収する。"""
    try:
        return wait_for_process(process, timeout_seconds=timeout_seconds)
    except subprocess.TimeoutExpired as error:
        raise MediaRuntimeError(
            MediaRuntimeFailureReason.DECODER_STALLED,
            "Video ScanのFFmpeg decode進捗が停止しました",
        ) from error


def _terminate_and_reap_scan_process(process: subprocess.Popen[str]) -> None:
    """Video Scan processを猶予付きで終了し必ず子processとして回収する。"""
    with suppress(ProcessLookupError):
        os.kill(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=_VIDEO_SCAN_TERMINATION_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.kill(process.pid, signal.SIGKILL)
        process.wait()


def _bounded_scale_filter(max_dimension: int) -> str:
    return (
        f"scale=w='min(iw,{max_dimension})':h='min(ih,{max_dimension})':"
        "force_original_aspect_ratio=decrease:force_divisible_by=2"
    )


def _remove_scan_proxy_sentinel(folder: Path) -> tuple[Path, ...]:
    """空signalも成功させる先頭sentinelを削除し実proxyだけを返す。"""
    files = tuple(sorted(folder.glob("*.jpg")))
    if not files:
        raise MediaRuntimeError(
            MediaRuntimeFailureReason.DECODER_FAILURE,
            "Video Scanのproxy sentinelが生成されませんでした",
        )
    files[0].unlink()
    return files[1:]


def _pcm_range_timing_is_continuous(
    observed_timing: list[tuple[int, int]],
    expected_start_pts: int,
    expected_sample_count: int,
) -> bool:
    """小さなpacket量子化だけを許しrange内のsample gridを検証する。"""
    observed_sample_count = 0
    for pts, sample_count in observed_timing:
        expected_pts = expected_start_pts + observed_sample_count
        if (
            sample_count < 1
            or abs(pts - expected_pts) > _PCM_PTS_QUANTIZATION_TOLERANCE_SAMPLES
        ):
            return False
        observed_sample_count += sample_count
    return observed_sample_count == expected_sample_count


def _ffmpeg_number(value: float) -> str:
    return format(value, ".15g")


def _validate_scan_configuration(
    heartbeat_interval_seconds: float,
    scene_change_threshold: float,
    scene_min_interval_seconds: float,
    decode_backend: str,
) -> None:
    values = (
        heartbeat_interval_seconds,
        scene_change_threshold,
        scene_min_interval_seconds,
    )
    if (
        any(not isinstance(value, int | float) for value in values)
        or any(not float("-inf") < value < float("inf") for value in values)
        or heartbeat_interval_seconds <= 0
        or not 0 <= scene_change_threshold <= 1
        or scene_min_interval_seconds <= 0
        or decode_backend not in {"cpu", "nvdec"}
    ):
        msg = "Video Scan設定が不正です"
        raise ValueError(msg)


def _parse_named_showinfo(
    line: str,
) -> tuple[str, tuple[int, int | None, int, int]] | None:
    if " n:" not in line:
        return None
    branch_match = _SHOWINFO_BRANCH_PATTERN.search(line)
    pts_match = _SHOWINFO_PTS_PATTERN.search(line)
    size_match = _SHOWINFO_SIZE_PATTERN.search(line)
    if branch_match is None or pts_match is None or size_match is None:
        return None
    duration_match = _SHOWINFO_DURATION_PATTERN.search(line)
    duration = int(duration_match.group(1)) if duration_match is not None else None
    return (
        branch_match.group("branch"),
        (
            int(pts_match.group(1)),
            duration,
            int(size_match.group(1)),
            int(size_match.group(2)),
        ),
    )


def _scanned_frames(
    metadata: list[tuple[int, int | None, int, int]],
    paths: tuple[Path, ...],
    time_base: Fraction,
) -> list[ScannedVideoFrame]:
    return [
        ScannedVideoFrame(
            source_pts=pts,
            duration_ts=duration,
            time_base=time_base,
            width=width,
            height=height,
            image_path=path,
        )
        for (pts, duration, width, height), path in zip(
            metadata,
            paths,
            strict=True,
        )
    ]
