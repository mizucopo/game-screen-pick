"""exactなVideo Duration。"""

from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class VideoDuration:
    """0から動画終端までの正の既約分数秒。"""

    seconds: Fraction

    def __post_init__(self) -> None:
        """正のdurationだけを受理する。"""
        if self.seconds <= 0:
            msg = "Video Durationは正である必要があります"
            raise ValueError(msg)
