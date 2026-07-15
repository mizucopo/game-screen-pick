"""system FFmpeg/ffprobeをsemantic media operationへ閉じ込めるruntime。"""

import json
import re
import resource
import subprocess
import time
from collections import deque
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path

from ..models.decoded_video_frame import DecodedVideoFrame
from ..models.embedded_subtitle import EmbeddedSubtitle
from ..models.media_probe import MediaProbe
from ..models.media_runtime_error import MediaRuntimeError
from ..models.media_runtime_failure_reason import MediaRuntimeFailureReason
from ..models.media_runtime_identity import MediaRuntimeIdentity
from ..models.media_stream import MediaStream
from ..models.native_video_scan import NativeVideoScan
from ..models.pcm_audio_chunk import PcmAudioChunk
from ..models.scanned_video_frame import ScannedVideoFrame
from .ffmpeg_pcm_reader import iter_pcm_audio_chunks
from .ffmpeg_subtitle_reader import read_embedded_subtitle_events
from .ffmpeg_video_reader import iter_decoded_video_frames
from .ffprobe_parser import parse_media_probe

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
        "ashowinfo",
        "format",
        "nullsink",
        "scale",
        "select",
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
_SHOWINFO_BRANCH_PATTERN = re.compile(r"showinfo@(?P<branch>timeline|heartbeat|scene)")
_SHOWINFO_PTS_PATTERN = re.compile(r"\bpts:\s*(-?\d+)")
_SHOWINFO_DURATION_PATTERN = re.compile(r"\bduration:\s*(-?\d+)")
_SHOWINFO_SIZE_PATTERN = re.compile(r"\bs:(\d+)x(\d+)")


class FfmpegMediaRuntime:
    """PATH上のsystem FFmpegとffprobeを使うmedia runtime。"""

    def __init__(
        self,
        ffmpeg_executable: str = "ffmpeg",
        ffprobe_executable: str = "ffprobe",
    ) -> None:
        self._ffmpeg_executable = ffmpeg_executable
        self._ffprobe_executable = ffprobe_executable

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
        self._verify_capabilities()
        return MediaRuntimeIdentity(
            ffmpeg_version=ffmpeg_version,
            ffprobe_version=ffprobe_version,
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
        _validate_scan_configuration(
            heartbeat_interval_seconds,
            scene_change_threshold,
            scene_min_interval_seconds,
            decode_backend,
        )
        if stream.kind != "video" or stream.time_base is None:
            msg = "Video Scanにはexact time baseを持つvideo streamが必要です"
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
        )
        usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
        started_at = time.monotonic()
        timeline_first: tuple[int, int | None, int, int] | None = None
        timeline_last: tuple[int, int | None, int, int] | None = None
        heartbeat_metadata: list[tuple[int, int | None, int, int]] = []
        scene_metadata: list[tuple[int, int | None, int, int]] = []
        stderr_tail: deque[str] = deque(maxlen=80)
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            if process.stderr is None:
                msg = "FFmpeg scan stderrを開始できません"
                raise RuntimeError(msg)
            for line in process.stderr:
                stderr_tail.append(line.rstrip())
                parsed = _parse_named_showinfo(line)
                if parsed is None:
                    continue
                branch, metadata = parsed
                if branch == "timeline":
                    if timeline_first is None:
                        timeline_first = metadata
                    timeline_last = metadata
                elif branch == "heartbeat":
                    heartbeat_metadata.append(metadata)
                else:
                    scene_metadata.append(metadata)
            return_code = process.wait()
        except OSError as error:
            raise MediaRuntimeError(
                MediaRuntimeFailureReason.DECODER_FAILURE,
                "Video ScanのFFmpeg processを開始できませんでした",
            ) from error
        wall_seconds = time.monotonic() - started_at
        usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu_seconds = (
            usage_after.ru_utime
            - usage_before.ru_utime
            + usage_after.ru_stime
            - usage_before.ru_stime
        )
        if return_code != 0:
            detail = "\n".join(stderr_tail)
            raise MediaRuntimeError(
                MediaRuntimeFailureReason.DECODER_FAILURE,
                f"Video ScanのFFmpeg decodeに失敗しました\n{detail}",
            )
        if timeline_first is None or timeline_last is None:
            raise MediaRuntimeError(
                MediaRuntimeFailureReason.DECODER_FAILURE,
                "Video Scanに表示可能frameがありません",
            )
        heartbeat_files = tuple(sorted(heartbeat_folder.glob("*.jpg")))
        scene_files = tuple(sorted(scene_folder.glob("*.jpg")))
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
        scenes_with_dummy = _scanned_frames(
            scene_metadata,
            scene_files,
            stream.time_base,
        )
        if not scenes_with_dummy:
            raise MediaRuntimeError(
                MediaRuntimeFailureReason.DECODER_FAILURE,
                "Video Scanのscene分析枝を初期化できませんでした",
            )
        scenes_with_dummy[0].image_path.unlink()
        scene_frames = tuple(scenes_with_dummy[1:])
        origin_pts, _origin_duration, _origin_width, _origin_height = timeline_first
        last_pts, last_duration, _last_width, _last_height = timeline_last
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
        )

    def scan_video_frame_ranges(
        self,
        media_path: Path,
        stream_index: int,
        pts_ranges: tuple[tuple[int, int], ...],
        max_dimension: int,
    ) -> Iterator[DecodedVideoFrame]:
        """半開PTS rangeの和集合にあるnative RGB24 frameだけを返す。"""
        if (
            max_dimension < 1
            or not pts_ranges
            or any(start >= end for start, end in pts_ranges)
        ):
            msg = "PTS rangeとmax_dimensionが不正です"
            raise ValueError(msg)
        expressions = [
            f"gte(pts\\,{start})*lt(pts\\,{end})" for start, end in pts_ranges
        ]
        frame_filter = f"select='{'+'.join(expressions)}'," + _scale_filter(
            max_dimension
        )
        command = self._video_decode_command(
            media_path,
            stream_index,
            frame_filter,
        )
        try:
            yield from iter_decoded_video_frames(command, stream_index)
        except _DECODE_ERRORS as error:
            raise MediaRuntimeError(
                MediaRuntimeFailureReason.FRAME_EXTRACTION_FAILED,
                "指定されたPTS rangeのnative frameを抽出できませんでした",
            ) from error

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
        command = self._video_decode_command(
            media_path,
            stream_index,
            frame_filter,
            frame_limit=1,
        )
        try:
            frames = tuple(iter_decoded_video_frames(command, stream_index))
        except _DECODE_ERRORS as error:
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

    def write_mjpeg_proxy(
        self,
        frame: DecodedVideoFrame,
        output_path: Path,
        *,
        quality: int,
    ) -> None:
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
        try:
            subprocess.run(
                command,
                input=frame.pixels,
                check=True,
                capture_output=True,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise MediaRuntimeError(
                MediaRuntimeFailureReason.FRAME_EXTRACTION_FAILED,
                "Frame Candidate ProxyをMJPEGへencodeできませんでした",
            ) from error

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
            f"asetnsamples=n={frame_sample_count}:p=0,ashowinfo"
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
    ) -> list[str]:
        command = self._decode_command_prefix(media_path, stream_index)
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
    ) -> list[str]:
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
        command.extend(["-i", str(media_path)])
        heartbeat_interval = _ffmpeg_number(heartbeat_interval_seconds)
        scene_threshold = _ffmpeg_number(scene_change_threshold)
        scene_interval = _ffmpeg_number(scene_min_interval_seconds)
        graph = ";".join(
            (
                f"[0:{stream.index}]split=3[timeline_source]"
                "[heartbeat_source][scene_source]",
                "[timeline_source]showinfo@timeline,nullsink",
                (
                    "[heartbeat_source]"
                    "select='isnan(prev_selected_t)+"
                    f"gte(t-prev_selected_t,{heartbeat_interval})',"
                    f"{_bounded_scale_filter(960)},"
                    "format=yuvj420p,showinfo@heartbeat[heartbeat_output]"
                ),
                (
                    "[scene_source]"
                    f"{_bounded_scale_filter(320)},"
                    "select='eq(n,0)+gte(n,1)*"
                    f"gt(scene,{scene_threshold})*"
                    "(isnan(prev_selected_t)+"
                    f"gte(t-prev_selected_t,{scene_interval}))',"
                    "format=yuvj420p,showinfo@scene[scene_output]"
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

    def _verify_capabilities(self) -> None:
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
        if (
            not _REQUIRED_DEMUXERS.issubset(demuxers)
            or not _REQUIRED_DECODERS.issubset(decoders)
            or not _REQUIRED_ENCODERS.issubset(encoders)
            or not _REQUIRED_MUXERS.issubset(muxers)
            or not _REQUIRED_FILTERS.issubset(filters)
            or not isinstance(probe_document, dict)
            or not isinstance(probe_document.get("program_version"), dict)
        ):
            self._raise_missing_capability()

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
    def _raise_missing_capability(error: BaseException | None = None) -> None:
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


def _bounded_scale_filter(max_dimension: int) -> str:
    return (
        f"scale=w='min(iw,{max_dimension})':h='min(ih,{max_dimension})':"
        "force_original_aspect_ratio=decrease:force_divisible_by=2"
    )


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
