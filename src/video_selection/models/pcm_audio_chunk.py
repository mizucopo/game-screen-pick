"""連続decodeされたPCM sample gridの一部分。"""

from dataclasses import dataclass
from fractions import Fraction
from typing import Literal


@dataclass(frozen=True)
class PcmAudioChunk:
    """exact output PTSとsample位置を持つmono s16le PCM chunk。"""

    stream_index: int
    sample_start: int
    sample_count: int
    sample_rate: int
    channel_count: Literal[1]
    sample_format: Literal["s16le"]
    pts: int
    time_base: Fraction
    pcm_bytes: bytes

    def __post_init__(self) -> None:
        """sample gridとPCM artifactの整合を検証する。"""
        if (
            self.sample_start < 0
            or self.sample_count <= 0
            or self.sample_rate <= 0
            or self.time_base <= 0
        ):
            msg = "PCM Audio Chunkのsample gridが不正です"
            raise ValueError(msg)
        if len(self.pcm_bytes) != self.sample_count * 2:
            msg = "PCM Audio Chunkのbyte数がsample数と一致しません"
            raise ValueError(msg)
