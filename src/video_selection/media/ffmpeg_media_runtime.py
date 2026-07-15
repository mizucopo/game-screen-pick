"""system FFmpeg/ffprobeをsemantic media operationへ閉じ込めるruntime。"""

import json
import re
import subprocess
from collections.abc import Iterator
from pathlib import Path

from ..models.decoded_video_frame import DecodedVideoFrame
from ..models.embedded_subtitle import EmbeddedSubtitle
from ..models.media_probe import MediaProbe
from ..models.media_runtime_error import MediaRuntimeError
from ..models.media_runtime_failure_reason import MediaRuntimeFailureReason
from ..models.media_runtime_identity import MediaRuntimeIdentity
from ..models.pcm_audio_chunk import PcmAudioChunk
from .ffmpeg_pcm_reader import iter_pcm_audio_chunks
from .ffmpeg_subtitle_reader import read_embedded_subtitle_events
from .ffmpeg_video_reader import iter_decoded_video_frames
from .ffprobe_parser import parse_media_probe

_VERSION_PATTERN = re.compile(r"^(?:ffmpeg|ffprobe) version (?P<version>\S+)")
_SEMANTIC_VERSION_PATTERN = re.compile(
    r"(?P<major>\d+)\.(?P<minor>\d+)(?:\.(?P<patch>\d+))?"
)
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
_REQUIRED_FILTERS = frozenset(
    {
        "aformat",
        "aresample",
        "asetnsamples",
        "ashowinfo",
        "format",
        "scale",
        "select",
        "showinfo",
    }
)
_DECODE_ERRORS = (
    OSError,
    subprocess.CalledProcessError,
    EOFError,
    ValueError,
)


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
        except OSError as error:
            msg = f"{missing_reason.value}"
            raise MediaRuntimeError(missing_reason, msg) from error
        first_line = completed.stdout.splitlines()[0]
        match = _VERSION_PATTERN.match(first_line)
        if match is None:
            msg = "FFmpeg toolのversionを解決できません"
            raise RuntimeError(msg)
        build_signature = tuple(
            line.strip()
            for line in completed.stdout.splitlines()[1:]
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
            or not _REQUIRED_FILTERS.issubset(filters)
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
            if len(fields) < 2 or not set(fields[0]) <= set(".ADEILNPRSTV"):
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
