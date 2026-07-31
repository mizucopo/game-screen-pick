"""ashowinfo付きraw pipeをstreaming PCM Audio Chunkへ変換する。"""

import queue
import re
import subprocess
import threading
from collections.abc import Iterator
from fractions import Fraction
from typing import IO, TypeAlias

from ..models.pcm_audio_chunk import PcmAudioChunk

_PTS_PATTERN = re.compile(r"\bpts:(-?\d+)")
_SAMPLE_COUNT_PATTERN = re.compile(r"\bnb_samples:(\d+)")

_AudioMetadata: TypeAlias = tuple[int, int]
_MetadataItem: TypeAlias = _AudioMetadata | BaseException | None


def iter_pcm_audio_chunks(
    command: list[str],
    stream_index: int,
    sample_rate: int,
    *,
    initial_sample_start: int = 0,
) -> Iterator[PcmAudioChunk]:
    """一つのFFmpeg processから連続PCM chunkを順次返す。"""
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdout is None or process.stderr is None:
        msg = "FFmpeg audio pipeを開始できません"
        raise RuntimeError(msg)
    metadata_queue: queue.Queue[_MetadataItem] = queue.Queue()
    stderr_thread = threading.Thread(
        target=_collect_ashowinfo,
        args=(process.stderr, metadata_queue),
        daemon=True,
    )
    stderr_thread.start()
    try:
        yield from _read_chunks(
            process.stdout,
            metadata_queue,
            stream_index,
            sample_rate,
            initial_sample_start,
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


def _collect_ashowinfo(
    stderr: IO[bytes],
    metadata_queue: queue.Queue[_MetadataItem],
) -> None:
    try:
        for raw_line in stderr:
            line = raw_line.decode("utf-8", errors="replace")
            if "Parsed_ashowinfo" not in line or " n:" not in line:
                continue
            pts_match = _PTS_PATTERN.search(line)
            sample_count_match = _SAMPLE_COUNT_PATTERN.search(line)
            if pts_match is None or sample_count_match is None:
                msg = "ashowinfo metadataが不正です"
                raise ValueError(msg)
            metadata_queue.put(
                (int(pts_match.group(1)), int(sample_count_match.group(1)))
            )
    except BaseException as error:
        metadata_queue.put(error)
    finally:
        metadata_queue.put(None)


def _read_chunks(
    stdout: IO[bytes],
    metadata_queue: queue.Queue[_MetadataItem],
    stream_index: int,
    sample_rate: int,
    initial_sample_start: int,
) -> Iterator[PcmAudioChunk]:
    sample_start = initial_sample_start
    while True:
        metadata = metadata_queue.get()
        if isinstance(metadata, BaseException):
            raise metadata
        if metadata is None:
            return
        pts, sample_count = metadata
        pcm_bytes = _read_exact(stdout, sample_count * 2)
        yield PcmAudioChunk(
            stream_index=stream_index,
            sample_start=sample_start,
            sample_count=sample_count,
            sample_rate=sample_rate,
            channel_count=1,
            sample_format="s16le",
            pts=pts,
            time_base=Fraction(1, sample_rate),
            pcm_bytes=pcm_bytes,
        )
        sample_start += sample_count


def _read_exact(stream: IO[bytes], byte_count: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < byte_count:
        chunk = stream.read(byte_count - len(chunks))
        if not chunk:
            msg = "FFmpeg PCM artifactが途中で終了しました"
            raise EOFError(msg)
        chunks.extend(chunk)
    return bytes(chunks)
