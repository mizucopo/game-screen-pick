"""Video Input Folder内の対応video path探索。"""

import re
from collections.abc import Iterator
from pathlib import Path

SUPPORTED_VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".mkv", ".webm"})


def discover_video_paths(input_folder: Path, recursive: bool) -> tuple[Path, ...]:
    """symlink境界を守り対応videoを相対path自然順で返す。"""
    candidates = tuple(_iter_video_paths(input_folder, recursive))
    return tuple(
        sorted(
            candidates,
            key=lambda path: _natural_relative_path_key(path, input_folder),
        )
    )


def _iter_video_paths(input_folder: Path, recursive: bool) -> Iterator[Path]:
    directories = [input_folder]
    while directories:
        directory = directories.pop()
        for path in directory.iterdir():
            if directory == input_folder and path.name == ".game-screen-pick":
                continue
            if path.is_symlink():
                if _is_supported_video_file(path):
                    yield path
                continue
            if path.is_dir():
                if recursive:
                    directories.append(path)
                continue
            if _is_supported_video_file(path):
                yield path


def _is_supported_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.casefold() in SUPPORTED_VIDEO_SUFFIXES


def _natural_relative_path_key(
    path: Path,
    input_folder: Path,
) -> tuple[tuple[tuple[int, int | str], ...], str]:
    relative_path = path.relative_to(input_folder).as_posix()
    natural_parts = tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", relative_path)
        if part
    )
    return natural_parts, relative_path
