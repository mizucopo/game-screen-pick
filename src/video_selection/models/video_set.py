"""一回の選定で扱うVideo Set。"""

from dataclasses import dataclass
from pathlib import Path

from .video_source import VideoSource


@dataclass(frozen=True)
class VideoSet:
    """Video Input Folderから発見された順序付き不変snapshot。"""

    input_folder: Path
    sources: tuple[VideoSource, ...]
    fingerprint: str
    recursive: bool

    def __post_init__(self) -> None:
        """空集合と不正なVideo Set Fingerprintを拒否する。"""
        if not self.sources:
            msg = "Video Setには1本以上のVideo Sourceが必要です"
            raise ValueError(msg)
        if len(self.fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in self.fingerprint
        ):
            msg = "Video Set Fingerprintには64桁のSHA-256が必要です"
            raise ValueError(msg)

    @property
    def videos(self) -> tuple[Path, ...]:
        """Video Orderを保った実filesystem pathを返す。"""
        return tuple(source.path for source in self.sources)

    @property
    def relative_paths(self) -> tuple[str, ...]:
        """Video Orderを保った入力rootからの相対pathを返す。"""
        return tuple(source.relative_path for source in self.sources)
