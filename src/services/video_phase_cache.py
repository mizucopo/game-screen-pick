"""Input Video Directory配下へ保存するphase別動画cache."""

from __future__ import annotations

import stat
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from ..utils.video_selection_files import (
    json_digest,
    read_json,
    write_json_atomic,
    write_text_atomic,
)

CACHE_DIRECTORY_NAME = "cache-game-screen-pick"
CACHE_INFO_FILENAME = "CACHE_INFO.txt"
CACHE_SCHEMA_VERSION = 1
VIDEO_IDENTITY_VERSION = 1

_CACHE_INFO = """game-screen-pick 再開cache

このfolderには、親Input Video Directory内の動画から再生成できる処理cacheを保存します。
game-screen-pickを実行していないときは、cache-game-screen-pick folder全体を安全に
削除できます。必要なcacheは次回実行時に再生成されます。Selected Image、
Selected Contact Sheet、report.jsonはここではなく、指定したOutput Folderへ保存します。

Input Videoは相対ファイル名とfile sizeで識別します。SHA-256、mtime、絶対pathは
run間の同一性判定へ使用しません。同じ相対ファイル名とsizeを保ったまま内容を変更し、
再生成を強制したい場合は、このfolderを削除してください。
"""


@dataclass(frozen=True)
class VideoCacheIdentity:
    """cache上でInput Videoを識別するpath非依存の値."""

    relative_path: str
    size: int
    key: str


def prepare_cache_directory(cache_root: Path, directory: Path) -> Path:
    """cache root配下のsymlinkを辿らずmanaged directoryを用意する."""
    try:
        relative = directory.relative_to(cache_root)
    except ValueError as error:
        raise RuntimeError(f"cache directoryがcache root外です: {directory}") from error
    current = cache_root
    for part in ("", *relative.parts):
        if part:
            current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            with suppress(FileExistsError):
                current.mkdir()
            mode = current.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise RuntimeError(f"cache directoryにsymlinkは使用できません: {current}")
        if not stat.S_ISDIR(mode):
            raise RuntimeError(f"cache pathがdirectoryではありません: {current}")
    return directory


def resolve_input_directory(videos: tuple[Path, ...]) -> Path:
    """全Input Videoが直下にある共通Input Video Directoryを返す."""
    if not videos:
        raise ValueError("入力動画を1本以上指定してください")
    input_directory = videos[0].parent
    for video in videos:
        if video.parent != input_directory:
            raise ValueError(
                "すべての入力動画は同じInput Video Directory直下に必要です"
            )
    return input_directory


def build_video_identity(
    input_directory: Path,
    video: Path,
) -> VideoCacheIdentity:
    """相対ファイル名とsizeだけからInput Video identityを作る."""
    try:
        relative = video.relative_to(input_directory)
    except ValueError as error:
        raise ValueError("入力動画がInput Video Directory外にあります") from error
    if len(relative.parts) != 1:
        raise ValueError("入力動画はInput Video Directory直下に必要です")
    size = video.stat().st_size
    relative_path = relative.as_posix()
    key = json_digest(
        {
            "identity_version": VIDEO_IDENTITY_VERSION,
            "relative_path": relative_path,
            "size": size,
        }
    )
    return VideoCacheIdentity(relative_path=relative_path, size=size, key=key)


def prepare_cache_root(input_directory: Path) -> Path:
    """見えるcache rootと削除可能性を説明するfileを用意する."""
    cache_root = input_directory / CACHE_DIRECTORY_NAME
    if cache_root.is_symlink():
        raise RuntimeError(f"cache directoryにsymlinkは使用できません: {cache_root}")
    cache_root.mkdir(parents=True, exist_ok=True)
    info_path = cache_root / CACHE_INFO_FILENAME
    if info_path.is_symlink():
        raise RuntimeError(f"cache説明fileにsymlinkは使用できません: {info_path}")
    try:
        current_info = info_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        current_info = None
    if current_info != _CACHE_INFO:
        write_text_atomic(info_path, _CACHE_INFO)
    return cache_root


def phase_key(
    phase: str,
    phase_version: int,
    conditions: dict[str, Any],
) -> str:
    """phase versionとsemantic inputからcache keyを作る."""
    return json_digest(
        {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "phase": phase,
            "phase_version": phase_version,
            "conditions": conditions,
        }
    )


def read_phase_data(
    path: Path,
    *,
    phase: str,
    phase_version: int,
    expected_key: str,
) -> dict[str, Any] | None:
    """完全に一致する正常なphase cacheだけを返す."""
    if not path.is_file() or path.is_symlink():
        return None
    try:
        payload = read_json(path)
    except (OSError, ValueError):
        return None
    if (
        not isinstance(payload, dict)
        or payload.get("cache_schema_version") != CACHE_SCHEMA_VERSION
        or payload.get("phase") != phase
        or payload.get("phase_version") != phase_version
        or payload.get("cache_key") != expected_key
        or not isinstance(payload.get("data"), dict)
    ):
        return None
    return cast(dict[str, Any], payload["data"])


def write_phase_data(
    path: Path,
    *,
    phase: str,
    phase_version: int,
    cache_key: str,
    data: dict[str, Any],
) -> None:
    """phase cacheを共通envelopeでatomic保存する."""
    write_json_atomic(
        path,
        {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "phase": phase,
            "phase_version": phase_version,
            "cache_key": cache_key,
            "data": data,
        },
    )


def stable_frame_id(video_key: str, sample_index: int) -> str:
    """入力集合内の位置に依存しない数字だけのframe IDを返す."""
    if sample_index <= 0:
        raise ValueError("sample indexは正の整数で指定してください")
    source_number = int(video_key[:16], 16)
    return f"f{source_number:020d}{sample_index:05d}"
