"""一回の選定で扱うVideo Set。"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoSet:
    """Video Input Folderから発見された順序付きvideo集合。"""

    input_folder: Path
    videos: tuple[Path, ...]

    @property
    def relative_paths(self) -> tuple[str, ...]:
        """Video Orderを保った入力rootからの相対pathを返す。"""
        return tuple(
            video.relative_to(self.input_folder).as_posix() for video in self.videos
        )
