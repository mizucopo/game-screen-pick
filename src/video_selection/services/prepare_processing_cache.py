"""Input Lock内で行うprocessing cache準備。"""

import shutil
from pathlib import Path
from uuid import uuid4

from ..models.legacy_cache_cleanup_diagnostic import LegacyCacheCleanupDiagnostic
from .input_folder_lock import InputFolderLock


def prepare_processing_cache(
    cache_folder: Path,
    *,
    input_lock: InputFolderLock,
    reset_cache: bool,
) -> LegacyCacheCleanupDiagnostic:
    """書込検査後にresetまたは認識済みlegacy削除を実行する。"""
    input_lock.assert_held_for(cache_folder)
    _require_safe_cache_folder(cache_folder)
    cache_folder.mkdir(parents=True, exist_ok=True)
    _verify_cache_writable(cache_folder)
    if reset_cache:
        shutil.rmtree(cache_folder)
        cache_folder.mkdir()
        return LegacyCacheCleanupDiagnostic(removed_entry_count=0, removed_bytes=0)
    return _remove_recognized_legacy_cache(cache_folder)


def _require_safe_cache_folder(cache_folder: Path) -> None:
    if cache_folder.is_symlink() or (
        cache_folder.exists() and not cache_folder.is_dir()
    ):
        msg = "processing cache rootには通常directoryが必要です"
        raise RuntimeError(msg)


def _verify_cache_writable(cache_folder: Path) -> None:
    probe_path = cache_folder / f".write-probe-{uuid4().hex}"
    try:
        probe_path.write_bytes(b"")
    finally:
        probe_path.unlink(missing_ok=True)


def _remove_recognized_legacy_cache(
    cache_folder: Path,
) -> LegacyCacheCleanupDiagnostic:
    removed_entry_count = 0
    removed_bytes = 0
    neutral_analysis = cache_folder / "neutral-analysis"
    if neutral_analysis.is_dir() and not neutral_analysis.is_symlink():
        removed_bytes += _directory_content_bytes(neutral_analysis)
        shutil.rmtree(neutral_analysis)
        removed_entry_count += 1

    ollama_scenes = cache_folder / "ollama-scenes.json"
    if ollama_scenes.is_file() and not ollama_scenes.is_symlink():
        removed_bytes += ollama_scenes.stat().st_size
        ollama_scenes.unlink()
        removed_entry_count += 1
    return LegacyCacheCleanupDiagnostic(
        removed_entry_count=removed_entry_count,
        removed_bytes=removed_bytes,
    )


def _directory_content_bytes(directory: Path) -> int:
    return sum(
        path.stat().st_size
        for path in directory.rglob("*")
        if path.is_file() and not path.is_symlink()
    )
