"""source packet timingとFFmpeg decoded subtitle textを結合する。"""

import json
import re
import subprocess
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path

from ..models.embedded_subtitle import EmbeddedSubtitle

_SRT_BLOCK_PATTERN = re.compile(
    r"(?:\A|\n\n)\d+\n[^\n]+\s+-->\s+[^\n]+\n(?P<text>.*?)(?=\n\n|\Z)",
    re.DOTALL,
)


def read_embedded_subtitle_events(
    ffmpeg_executable: str,
    ffprobe_executable: str,
    media_path: Path,
    stream_index: int,
    time_base: Fraction,
) -> tuple[EmbeddedSubtitle, ...]:
    """選択streamの元packet timingとdecoded textを順序で結合する。"""
    packets = _read_packets(ffprobe_executable, media_path, stream_index)
    texts = _decode_texts(ffmpeg_executable, media_path, stream_index)
    if len(packets) != len(texts):
        msg = "subtitle packetとdecoded eventの件数が一致しません"
        raise ValueError(msg)
    return tuple(
        EmbeddedSubtitle(
            stream_index=stream_index,
            pts=pts,
            duration_ts=duration_ts,
            time_base=time_base,
            text=text,
        )
        for (pts, duration_ts), text in zip(packets, texts, strict=True)
    )


def _read_packets(
    ffprobe_executable: str,
    media_path: Path,
    stream_index: int,
) -> tuple[tuple[int, int], ...]:
    completed = subprocess.run(
        [
            ffprobe_executable,
            "-v",
            "error",
            "-select_streams",
            str(stream_index),
            "-show_packets",
            "-show_entries",
            "packet=pts,duration",
            "-of",
            "json",
            str(media_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    document = json.loads(completed.stdout)
    if not isinstance(document, Mapping):
        msg = "subtitle packet resultが不正です"
        raise ValueError(msg)
    raw_packets = document.get("packets")
    if not isinstance(raw_packets, list):
        msg = "subtitle packet listがありません"
        raise ValueError(msg)
    packets: list[tuple[int, int]] = []
    for raw_packet in raw_packets:
        if not isinstance(raw_packet, Mapping):
            msg = "subtitle packetが不正です"
            raise ValueError(msg)
        packets.append(
            (
                _required_int(raw_packet.get("pts")),
                _required_int(raw_packet.get("duration")),
            )
        )
    return tuple(packets)


def _decode_texts(
    ffmpeg_executable: str,
    media_path: Path,
    stream_index: int,
) -> tuple[str, ...]:
    completed = subprocess.run(
        [
            ffmpeg_executable,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-xerror",
            "-copyts",
            "-i",
            str(media_path),
            "-map",
            f"0:{stream_index}",
            "-c:s",
            "srt",
            "-f",
            "srt",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    normalized = completed.stdout.replace("\r\n", "\n").strip()
    if not normalized:
        return ()
    return tuple(
        match.group("text").strip() for match in _SRT_BLOCK_PATTERN.finditer(normalized)
    )


def _required_int(value: object) -> int:
    if isinstance(value, bool):
        msg = "subtitle packet integerが不正です"
        raise ValueError(msg)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as error:
            msg = "subtitle packet integerが不正です"
            raise ValueError(msg) from error
    msg = "subtitle packet integerが不正です"
    raise ValueError(msg)
