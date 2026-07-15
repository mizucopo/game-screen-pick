"""Video Set選定のEffective Configuration。"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EffectiveConfiguration:
    """一回の内部Video Set選定に必要な最小設定。"""

    video_input_folder: Path
    output_folder: Path
    image_count: int

    def __post_init__(self) -> None:
        """要求画像枚数を検証する。"""
        if self.image_count < 1:
            msg = "image_countは正の整数である必要があります"
            raise ValueError(msg)

    @property
    def processing_cache_folder(self) -> Path:
        """Video Input Folderが所有するprocessing cacheを返す。"""
        return self.video_input_folder / ".game-screen-pick" / "cache"
