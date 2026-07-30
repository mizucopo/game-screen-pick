"""Video Input Folderからcontent-addressed Video Setを発見する。"""

import hashlib
from pathlib import Path

from ..configuration.configuration_error import ConfigurationError
from ..models.video_set import VideoSet
from ..models.video_source import VideoSource
from .discover_video_paths import discover_video_paths
from .source_snapshot_signature import source_snapshot_signature
from .video_identity_cache import VideoIdentityCache


def discover_video_set(
    input_folder: Path,
    recursive: bool = False,
    *,
    identity_cache: VideoIdentityCache | None = None,
) -> VideoSet:
    """対応videoを自然順で発見し内容identityを確定する。"""
    if not input_folder.is_dir():
        msg = "Video Input Folderが存在しません"
        raise ConfigurationError("VIDEO_INPUT_FOLDER_NOT_FOUND", msg)
    video_paths = discover_video_paths(input_folder, recursive)
    if not video_paths:
        msg = "Video Input Folderに対応videoがありません"
        raise ConfigurationError("VIDEO_INPUT_FOLDER_EMPTY", msg)
    sources = tuple(
        _build_video_source(input_folder, path, identity_cache) for path in video_paths
    )
    _reject_duplicate_videos(sources)
    return VideoSet(
        input_folder=input_folder,
        sources=sources,
        fingerprint=_build_video_set_fingerprint(sources),
        recursive=recursive,
    )


def _build_video_source(
    input_folder: Path,
    video_path: Path,
    identity_cache: VideoIdentityCache | None,
) -> VideoSource:
    """whole-file SHA-256と発見時statを一つのVideo Sourceにする。"""
    if identity_cache is None:
        before_stat = video_path.stat()
        with video_path.open("rb") as video_file:
            fingerprint = hashlib.file_digest(video_file, "sha256").hexdigest()
        after_stat = video_path.stat()
        if source_snapshot_signature(before_stat) != source_snapshot_signature(
            after_stat
        ):
            msg = "Video Set snapshotがfingerprint計算中に変更されました"
            raise ValueError(msg)
    else:
        fingerprint, after_stat, _reused = identity_cache.resolve(
            input_folder,
            video_path,
        )
    return VideoSource(
        path=video_path,
        relative_path=video_path.relative_to(input_folder).as_posix(),
        fingerprint=fingerprint,
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
            raise ConfigurationError("DUPLICATE_VIDEO", msg)


def _build_video_set_fingerprint(sources: tuple[VideoSource, ...]) -> str:
    fingerprint = hashlib.sha256()
    fingerprint.update(b"game-screen-pick/video-set-fingerprint@1\0")
    for source in sources:
        fingerprint.update(bytes.fromhex(source.fingerprint))
    return fingerprint.hexdigest()
