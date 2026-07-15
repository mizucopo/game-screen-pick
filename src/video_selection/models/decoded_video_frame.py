"""一回のFFmpeg decodeから得たsource frame。"""

from dataclasses import dataclass
from fractions import Fraction
from typing import Literal


@dataclass(frozen=True)
class DecodedVideoFrame:
    """exact PTSとRGB24 pixelを持つdecoded frame。"""

    stream_index: int
    pts: int
    duration_ts: int | None
    time_base: Fraction
    width: int
    height: int
    pixel_format: Literal["rgb24"]
    pixels: bytes

    def __post_init__(self) -> None:
        """time base、寸法、pixel artifactの整合を検証する。"""
        if self.time_base <= 0 or self.width <= 0 or self.height <= 0:
            msg = "Decoded Video Frameのtime baseと寸法は正である必要があります"
            raise ValueError(msg)
        if len(self.pixels) != self.width * self.height * 3:
            msg = "Decoded Video FrameのRGB24 byte数が寸法と一致しません"
            raise ValueError(msg)
