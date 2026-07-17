"""実行中Video Set snapshotの不変性検証。"""

import hashlib
import os
from pathlib import Path
from typing import NoReturn

from ..models.video_set import VideoSet
from ..models.video_source import VideoSource
from .discover_video_paths import discover_video_paths


def validate_video_set_snapshot(video_set: VideoSet) -> None:
    """相対path列、stat、contentが発見時から不変か検証する。"""
    for source, current_path in _validate_paths_and_stats(video_set):
        _validate_source_content(source, current_path)


def validate_video_set_snapshot_metadata(video_set: VideoSet) -> None:
    """相対path列とstatが発見時から不変か軽量に検証する。"""
    _validate_paths_and_stats(video_set)


def validate_video_source_snapshot(
    video_set: VideoSet,
    source: VideoSource,
) -> None:
    """全体path/statと対象Video Sourceのcontentを検証する。"""
    current_sources = _validate_paths_and_stats(video_set)
    for expected, current_path in current_sources:
        if expected == source:
            _validate_source_content(source, current_path)
            return
    _raise_snapshot_changed()


def _validate_paths_and_stats(
    video_set: VideoSet,
) -> tuple[tuple[VideoSource, Path], ...]:
    current_paths = discover_video_paths(video_set.input_folder, video_set.recursive)
    current_relative_paths = tuple(
        path.relative_to(video_set.input_folder).as_posix() for path in current_paths
    )
    if current_relative_paths != video_set.relative_paths:
        _raise_snapshot_changed()
    current_sources = tuple(zip(video_set.sources, current_paths, strict=True))
    for source, current_path in current_sources:
        if _stat_signature(current_path.stat()) != source.stat_signature:
            _raise_snapshot_changed()
    return current_sources


def _validate_source_content(source: VideoSource, current_path: Path) -> None:
    """一つのsourceをstat-content-statの順で照合する。"""
    before_stat = current_path.stat()
    if _stat_signature(before_stat) != source.stat_signature:
        _raise_snapshot_changed()
    with current_path.open("rb") as video_file:
        current_fingerprint = hashlib.file_digest(video_file, "sha256").hexdigest()
    after_stat = current_path.stat()
    if (
        _stat_signature(after_stat) != source.stat_signature
        or current_fingerprint != source.fingerprint
    ):
        _raise_snapshot_changed()


def _stat_signature(stat: os.stat_result) -> tuple[int, int, int, int]:
    return (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
    )


def _raise_snapshot_changed() -> NoReturn:
    msg = "Video Set snapshotが変更されました"
    raise ValueError(msg)
