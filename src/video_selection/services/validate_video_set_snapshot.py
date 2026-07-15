"""実行中Video Set snapshotの不変性検証。"""

import hashlib
import os

from ..models.video_set import VideoSet
from .discover_video_paths import discover_video_paths


def validate_video_set_snapshot(video_set: VideoSet) -> None:
    """相対path列、stat、contentが発見時から不変か検証する。"""
    current_paths = discover_video_paths(video_set.input_folder, video_set.recursive)
    current_relative_paths = tuple(
        path.relative_to(video_set.input_folder).as_posix() for path in current_paths
    )
    if current_relative_paths != video_set.relative_paths:
        _raise_snapshot_changed()
    for source, current_path in zip(video_set.sources, current_paths, strict=True):
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


def _raise_snapshot_changed() -> None:
    msg = "Video Set snapshotが変更されました"
    raise ValueError(msg)
