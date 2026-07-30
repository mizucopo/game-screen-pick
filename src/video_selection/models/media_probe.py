"""media container probeのsemantic result。"""

from dataclasses import dataclass
from fractions import Fraction

from .media_stream import MediaStream


@dataclass(frozen=True)
class MediaProbe:
    """container名とordered stream metadataを保持する。"""

    format_names: tuple[str, ...]
    streams: tuple[MediaStream, ...]
    duration: Fraction | None = None
