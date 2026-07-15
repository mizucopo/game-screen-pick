"""Video Input FolderからVideo Setを発見する。"""

import re
from pathlib import Path

from ..models.video_set import VideoSet

SUPPORTED_VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".mkv", ".webm"})


def discover_video_set(input_folder: Path) -> VideoSet:
    """root直下の対応videoを自然順で返す。"""
    if not input_folder.is_dir():
        msg = f"Video Input Folderが存在しません: {input_folder}"
        raise ValueError(msg)
    videos = tuple(
        sorted(
            (
                path
                for path in input_folder.iterdir()
                if path.is_file() and path.suffix.casefold() in SUPPORTED_VIDEO_SUFFIXES
            ),
            key=_natural_path_key,
        )
    )
    if not videos:
        msg = "Video Input Folderに対応videoがありません"
        raise ValueError(msg)
    return VideoSet(input_folder=input_folder, videos=videos)


def _natural_path_key(path: Path) -> tuple[tuple[str | int, ...], str]:
    """path名を数値部分込みの自然順keyへ変換する。"""
    return (
        tuple(
            int(part) if part.isdigit() else part.casefold()
            for part in re.split(r"(\d+)", path.name)
        ),
        path.name,
    )
