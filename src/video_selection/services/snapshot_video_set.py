"""Video Setのcontent snapshotを構築する。"""

import hashlib
from pathlib import Path

from ..models.video_set import VideoSet


def snapshot_video_set(video_set: VideoSet) -> tuple[dict[str, str], ...]:
    """Video Orderを保ったrelative pathとcontent digestを返す。"""
    return tuple(
        {
            "path": relative_path,
            "sha256": _file_sha256(video_path),
        }
        for video_path, relative_path in zip(
            video_set.videos,
            video_set.relative_paths,
            strict=True,
        )
    )


def _file_sha256(video_path: Path) -> str:
    """一つのvideo fileのSHA-256を返す。"""
    with open(video_path, "rb") as video_file:
        return hashlib.file_digest(video_file, "sha256").hexdigest()
