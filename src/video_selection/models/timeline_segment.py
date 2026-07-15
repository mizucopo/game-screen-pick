"""Video Source timelineの半開区間。"""

from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class TimelineSegment:
    """gapやoverlapなく並ぶ一つのexact時間区間。"""

    identifier: str
    start: Fraction
    end: Fraction

    def __post_init__(self) -> None:
        """IDと正の半開区間を検証する。"""
        if (
            not self.identifier.startswith("seg_")
            or len(self.identifier) != 68
            or any(
                character not in "0123456789abcdef" for character in self.identifier[4:]
            )
        ):
            msg = "Timeline Segment IDにはseg_と64桁SHA-256が必要です"
            raise ValueError(msg)
        if self.start < 0 or self.end <= self.start:
            msg = "Timeline Segmentには正しい半開区間が必要です"
            raise ValueError(msg)
