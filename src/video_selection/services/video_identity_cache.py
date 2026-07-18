"""path非依存filesystem snapshotからVideo content identityを再利用するcache。"""

import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

from ..models.video_source import VideoSource

_SCHEMA = "game-screen-pick/video-identity-cache@1.0.0"
_MAX_ENTRY_BYTES = 2048


class VideoIdentityCache:
    """安定したstat signatureに対応するwhole-file SHA-256を保持する。"""

    def __init__(self, processing_cache_folder: Path) -> None:
        self._root = processing_cache_folder / "video-identities"

    def lookup(self, stat: os.stat_result) -> str | None:
        """完全一致するstat signatureのcontent fingerprintを返す。"""
        signature = _stat_signature(stat)
        entry_path = self._entry_path(signature)
        try:
            if (
                self._root.parent.is_symlink()
                or self._root.is_symlink()
                or not entry_path.is_file()
                or entry_path.is_symlink()
                or entry_path.stat().st_size > _MAX_ENTRY_BYTES
            ):
                return None
            value = json.loads(entry_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        if (
            not isinstance(value, dict)
            or value.get("schema") != _SCHEMA
            or value.get("stat_signature") != list(signature)
        ):
            return None
        fingerprint = value.get("fingerprint")
        return fingerprint if _is_sha256(fingerprint) else None

    def store(self, source: VideoSource) -> None:
        """Video Sourceのidentityをpathを含めずatomicに保存する。"""
        if self._root.is_symlink() or (self._root.exists() and not self._root.is_dir()):
            msg = "Video Identity cacheには通常directoryが必要です"
            raise RuntimeError(msg)
        self._root.mkdir(parents=True, exist_ok=True)
        entry_path = self._entry_path(source.stat_signature)
        temporary_path = self._root / f".{entry_path.name}.{uuid4().hex}.tmp"
        payload = {
            "schema": _SCHEMA,
            "stat_signature": list(source.stat_signature),
            "fingerprint": source.fingerprint,
        }
        try:
            temporary_path.write_text(
                json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary_path, entry_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _entry_path(self, signature: tuple[int, int, int, int, int]) -> Path:
        canonical = json.dumps(signature, separators=(",", ":")).encode()
        key = hashlib.sha256(
            b"game-screen-pick/video-identity-stat@1\0" + canonical
        ).hexdigest()
        return self._root / f"{key}.json"


def _stat_signature(stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
