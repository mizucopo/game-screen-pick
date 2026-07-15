"""system media toolの解決済みidentity。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class MediaRuntimeIdentity:
    """同一buildとして検証されたFFmpegとffprobeのversion。"""

    ffmpeg_version: str
    ffprobe_version: str
