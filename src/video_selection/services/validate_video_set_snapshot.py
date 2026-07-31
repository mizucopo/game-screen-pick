"""実行中Video Set snapshotの不変性検証。"""

from pathlib import Path
from typing import NoReturn

from ..models.video_set import VideoSet
from ..models.video_source import VideoSource
from .discover_video_paths import discover_video_paths
from .source_snapshot_signature import source_snapshot_signature


def validate_video_set_snapshot(video_set: VideoSet) -> None:
    """相対path列、size、mtimeが発見時から不変か検証する。"""
    _validate_paths_and_stats(video_set)


def validate_video_set_snapshot_metadata(video_set: VideoSet) -> None:
    """相対path列とstatが発見時から不変か軽量に検証する。"""
    _validate_paths_and_stats(video_set)


def validate_video_source_snapshot(
    video_set: VideoSet,
    source: VideoSource,
) -> None:
    """全体path列と対象Video Sourceのsize、mtimeを検証する。"""
    current_sources = _validate_paths_and_stats(video_set)
    for expected, _current_path in current_sources:
        if expected == source:
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
        if source_snapshot_signature(current_path.stat()) != source.snapshot_signature:
            _raise_snapshot_changed()
    return current_sources


def _raise_snapshot_changed() -> NoReturn:
    msg = "Video Set snapshotが変更されました"
    raise ValueError(msg)
