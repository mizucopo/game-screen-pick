"""probeされた一つのmedia stream。"""

from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

MediaStreamKind = Literal["video", "audio", "subtitle", "data", "attachment"]


@dataclass(frozen=True)
class MediaStream:
    """FFmpeg固有JSONから正規化されたstream metadata。"""

    index: int
    kind: MediaStreamKind
    codec_name: str
    time_base: Fraction | None
    start_pts: int | None
    duration_ts: int | None
    width: int | None
    height: int | None
    sample_rate: int | None
    channels: int | None
    language: str | None
    is_default: bool
    is_forced: bool
    is_attached_picture: bool = False
