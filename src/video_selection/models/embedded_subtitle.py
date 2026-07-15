"""内蔵text subtitleのsemantic event。"""

from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class EmbeddedSubtitle:
    """元packet PTS/time baseとdecoded textを持つsubtitle event。"""

    stream_index: int
    pts: int
    duration_ts: int
    time_base: Fraction
    text: str

    def __post_init__(self) -> None:
        """正のduration/time baseと非空textを検証する。"""
        if self.duration_ts <= 0 or self.time_base <= 0 or not self.text.strip():
            msg = "Embedded Subtitle eventが不正です"
            raise ValueError(msg)
