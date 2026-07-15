"""showinfo付きimage pipeをstreaming Decoded Video Frameへ変換する。"""

import queue
import re
import subprocess
import threading
from collections.abc import Iterator
from fractions import Fraction
from typing import IO, TypeAlias

from ..models.decoded_video_frame import DecodedVideoFrame

_TIME_BASE_PATTERN = re.compile(r"config in time_base:\s*(\d+)/(\d+)")
_PTS_PATTERN = re.compile(r"\bpts:\s*(-?\d+)")
_DURATION_PATTERN = re.compile(r"\bduration:\s*(-?\d+)")
_SIZE_PATTERN = re.compile(r"\bs:(\d+)x(\d+)")

_FrameMetadata: TypeAlias = tuple[Fraction, int, int | None, int, int]
_MetadataItem: TypeAlias = _FrameMetadata | BaseException | None


def iter_decoded_video_frames(
    command: list[str],
    stream_index: int,
) -> Iterator[DecodedVideoFrame]:
    """一つのFFmpeg processからPTSとRGB24 frameを順次返す。"""
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdout is None or process.stderr is None:
        msg = "FFmpeg pipeを開始できません"
        raise RuntimeError(msg)
    metadata_queue: queue.Queue[_MetadataItem] = queue.Queue()
    stderr_thread = threading.Thread(
        target=_collect_showinfo,
        args=(process.stderr, metadata_queue),
        daemon=True,
    )
    stderr_thread.start()
    try:
        yield from _read_frames(
            process.stdout,
            metadata_queue,
            stream_index,
        )
        return_code = process.wait()
        stderr_thread.join()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()
        stderr_thread.join()
        process.stdout.close()
        process.stderr.close()


def _collect_showinfo(
    stderr: IO[bytes],
    metadata_queue: queue.Queue[_MetadataItem],
) -> None:
    time_base: Fraction | None = None
    try:
        for raw_line in stderr:
            line = raw_line.decode("utf-8", errors="replace")
            time_base_match = _TIME_BASE_PATTERN.search(line)
            if time_base_match is not None:
                time_base = Fraction(
                    int(time_base_match.group(1)),
                    int(time_base_match.group(2)),
                )
                continue
            if "Parsed_showinfo" not in line or " n:" not in line:
                continue
            if time_base is None:
                msg = "showinfo frameより前にtime baseがありません"
                raise ValueError(msg)
            pts_match = _PTS_PATTERN.search(line)
            duration_match = _DURATION_PATTERN.search(line)
            size_match = _SIZE_PATTERN.search(line)
            if pts_match is None or size_match is None:
                msg = "showinfo frame metadataが不正です"
                raise ValueError(msg)
            metadata_queue.put(
                (
                    time_base,
                    int(pts_match.group(1)),
                    int(duration_match.group(1))
                    if duration_match is not None
                    else None,
                    int(size_match.group(1)),
                    int(size_match.group(2)),
                )
            )
    except BaseException as error:
        metadata_queue.put(error)
    finally:
        metadata_queue.put(None)


def _read_frames(
    stdout: IO[bytes],
    metadata_queue: queue.Queue[_MetadataItem],
    stream_index: int,
) -> Iterator[DecodedVideoFrame]:
    while True:
        magic = _read_ppm_token(stdout)
        if magic is None:
            return
        if magic != b"P6":
            msg = "FFmpeg frame artifactがPPM P6ではありません"
            raise ValueError(msg)
        width = int(_require_ppm_token(stdout))
        height = int(_require_ppm_token(stdout))
        maximum = int(_require_ppm_token(stdout))
        if maximum != 255:
            msg = "FFmpeg frame artifactのRGB depthが不正です"
            raise ValueError(msg)
        pixels = _read_exact(stdout, width * height * 3)
        metadata = metadata_queue.get()
        if isinstance(metadata, BaseException):
            raise metadata
        if metadata is None:
            msg = "FFmpeg frameとshowinfo metadataの件数が一致しません"
            raise ValueError(msg)
        time_base, pts, duration_ts, metadata_width, metadata_height = metadata
        if (width, height) != (metadata_width, metadata_height):
            msg = "FFmpeg frameとshowinfo metadataの寸法が一致しません"
            raise ValueError(msg)
        yield DecodedVideoFrame(
            stream_index=stream_index,
            pts=pts,
            duration_ts=duration_ts,
            time_base=time_base,
            width=width,
            height=height,
            pixel_format="rgb24",
            pixels=pixels,
        )


def _read_ppm_token(stream: IO[bytes]) -> bytes | None:
    token = bytearray()
    while True:
        byte = stream.read(1)
        if not byte:
            return bytes(token) if token else None
        if byte == b"#" and not token:
            stream.readline()
            continue
        if byte.isspace():
            if token:
                return bytes(token)
            continue
        token.extend(byte)


def _require_ppm_token(stream: IO[bytes]) -> bytes:
    token = _read_ppm_token(stream)
    if token is None:
        msg = "FFmpeg frame artifactが途中で終了しました"
        raise EOFError(msg)
    return token


def _read_exact(stream: IO[bytes], byte_count: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < byte_count:
        chunk = stream.read(byte_count - len(chunks))
        if not chunk:
            msg = "FFmpeg frame pixelが途中で終了しました"
            raise EOFError(msg)
        chunks.extend(chunk)
    return bytes(chunks)
