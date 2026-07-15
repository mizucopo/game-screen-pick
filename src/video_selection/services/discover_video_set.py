"""Video Input Folderからcontent-addressed Video Setを発見する。"""

import hashlib
import os
from pathlib import Path

from ..models.video_set import VideoSet
from ..models.video_source import VideoSource
from .discover_video_paths import discover_video_paths


def discover_video_set(input_folder: Path, recursive: bool = False) -> VideoSet:
    """対応videoを自然順で発見し内容identityを確定する。"""
    if not input_folder.is_dir():
        msg = "Video Input Folderが存在しません"
        raise ValueError(msg)
    video_paths = discover_video_paths(input_folder, recursive)
    if not video_paths:
        msg = "Video Input Folderに対応videoがありません"
        raise ValueError(msg)
    sources = tuple(_build_video_source(input_folder, path) for path in video_paths)
    _reject_duplicate_videos(sources)
    return VideoSet(
        input_folder=input_folder,
        sources=sources,
        fingerprint=_build_video_set_fingerprint(sources),
        recursive=recursive,
    )


def _build_video_source(input_folder: Path, video_path: Path) -> VideoSource:
    """whole-file SHA-256と発見時statを一つのVideo Sourceにする。"""
    before_stat = video_path.stat()
    with video_path.open("rb") as video_file:
        fingerprint = hashlib.file_digest(video_file, "sha256").hexdigest()
    after_stat = video_path.stat()
    before_signature = _stat_signature(before_stat)
    after_signature = _stat_signature(after_stat)
    if before_signature != after_signature:
        msg = "Video Set snapshotがfingerprint計算中に変更されました"
        raise ValueError(msg)
    return VideoSource(
        path=video_path,
        relative_path=video_path.relative_to(input_folder).as_posix(),
        fingerprint=fingerprint,
        device=after_stat.st_dev,
        inode=after_stat.st_ino,
        size_bytes=after_stat.st_size,
        modified_at_ns=after_stat.st_mtime_ns,
    )


def _reject_duplicate_videos(sources: tuple[VideoSource, ...]) -> None:
    paths_by_fingerprint: dict[str, list[str]] = {}
    for source in sources:
        paths_by_fingerprint.setdefault(source.fingerprint, []).append(
            source.relative_path
        )
    for relative_paths in paths_by_fingerprint.values():
        if len(relative_paths) > 1:
            displayed_paths = ", ".join(relative_paths)
            msg = f"Duplicate Videoが見つかりました: {displayed_paths}"
            raise ValueError(msg)


def _build_video_set_fingerprint(sources: tuple[VideoSource, ...]) -> str:
    fingerprint = hashlib.sha256()
    fingerprint.update(b"game-screen-pick/video-set-fingerprint@1\0")
    for source in sources:
        fingerprint.update(bytes.fromhex(source.fingerprint))
    return fingerprint.hexdigest()


def _stat_signature(stat: os.stat_result) -> tuple[int, int, int, int]:
    return (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
    )
