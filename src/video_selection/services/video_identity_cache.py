"""Video content identityをsource単位で即時確定するdurable cache。"""

import hashlib
import json
import os
import shutil
import stat as stat_module
from datetime import datetime, timezone
from pathlib import Path
from typing import cast
from uuid import uuid4

from ..models.checkpoint_operation import CheckpointOperation
from ..models.progress_event import ProgressEvent
from ..protocols.run_observer import RunObserver
from .checkpoint_version import checkpoint_version
from .source_snapshot_signature import source_snapshot_signature

_SCHEMA = "game-screen-pick/video-identity-cache@2.0.0"
_DEFAULT_ENGINE_VERSION = checkpoint_version(CheckpointOperation.VIDEO_IDENTITY)
_MAX_ENTRY_BYTES = 2048


class VideoIdentityCache:
    """source照合、whole-file SHA-256、atomic checkpointを所有する。"""

    def __init__(
        self,
        cache_root: Path,
        *,
        engine_version: str = _DEFAULT_ENGINE_VERSION,
        observer: RunObserver | None = None,
    ) -> None:
        if not engine_version.strip():
            raise ValueError("Video Identity Engine versionが必要です")
        self._root = cache_root
        self._engine_version = engine_version
        self._observer = observer
        self._root_prepared = False

    @property
    def engine_version(self) -> str:
        """実際に使用するIdentity Engine versionを返す。"""
        return self._engine_version

    def resolve(
        self,
        input_folder: Path,
        video_path: Path,
    ) -> tuple[str, os.stat_result, bool]:
        """sourceのidentityを解決し、missなら計算直後にatomic保存する。"""
        self._prepare_root()
        logical_source_key = _logical_source_key(input_folder, video_path)
        before_stat = video_path.stat()
        cached = self._lookup(logical_source_key, before_stat)
        if cached is not None:
            fingerprint, work_unit_fingerprint = cached
            after_stat = video_path.stat()
            _validate_stable_snapshot(before_stat, after_stat)
            self._observe_resolution(work_unit_fingerprint, reused=True)
            return fingerprint, after_stat, True

        with video_path.open("rb") as video_file:
            fingerprint = hashlib.file_digest(video_file, "sha256").hexdigest()
        after_stat = video_path.stat()
        _validate_stable_snapshot(before_stat, after_stat)
        self._store(
            logical_source_key,
            after_stat,
            fingerprint,
        )
        self._observe_resolution(
            _work_unit_fingerprint(
                self._engine_version,
                logical_source_key,
                after_stat,
            ),
            reused=False,
        )
        return fingerprint, after_stat, False

    def reset(self) -> None:
        """明示reset時だけdurable identity cache全体を削除する。"""
        if self._root.is_symlink():
            raise RuntimeError("Video Identity cache rootにsymbolic linkは使えません")
        if self._root.exists() and not self._root.is_dir():
            raise RuntimeError("Video Identity cacheには通常directoryが必要です")
        self._root_prepared = False
        if self._root.exists():
            shutil.rmtree(self._root)

    def _lookup(
        self,
        logical_source_key: str,
        source_stat: os.stat_result,
    ) -> tuple[str, str] | None:
        """engine、logical source、size、mtimeが一致するentryだけを返す。"""
        entry_path = self._entry_path(logical_source_key)
        try:
            entry_stat = entry_path.lstat()
        except FileNotFoundError:
            return None
        if (
            stat_module.S_ISLNK(entry_stat.st_mode)
            or not stat_module.S_ISREG(entry_stat.st_mode)
            or entry_stat.st_size > _MAX_ENTRY_BYTES
        ):
            self._remove_invalid_entry(entry_path)
            return None
        try:
            value: object = json.loads(entry_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._remove_invalid_entry(entry_path)
            return None
        if not isinstance(value, dict) or not all(
            isinstance(key, str) for key in value
        ):
            self._remove_invalid_entry(entry_path)
            return None
        entry = cast(dict[str, object], value)
        fingerprint = entry.get("fingerprint")
        work_unit_fingerprint = entry.get("work_unit_fingerprint")
        completed_at = entry.get("completed_at")
        expected_work_unit_fingerprint = _work_unit_fingerprint(
            self._engine_version,
            logical_source_key,
            source_stat,
        )
        if (
            entry.get("schema") != _SCHEMA
            or entry.get("engine_version") != self._engine_version
            or entry.get("logical_source_key") != logical_source_key
            or entry.get("size_bytes") != source_stat.st_size
            or entry.get("modified_at_ns") != source_stat.st_mtime_ns
            or not _is_sha256(fingerprint)
            or work_unit_fingerprint != expected_work_unit_fingerprint
            or not _is_timezone_aware_iso_datetime(completed_at)
        ):
            self._remove_invalid_entry(entry_path)
            return None
        return cast(str, fingerprint), expected_work_unit_fingerprint

    def _store(
        self,
        logical_source_key: str,
        stat: os.stat_result,
        fingerprint: str,
    ) -> None:
        """一つのVideo Identityをpathなしでatomicに保存する。"""
        if not _is_sha256(fingerprint):
            raise ValueError("Video Fingerprintには完全なSHA-256が必要です")
        if not self._root_is_safe():
            raise RuntimeError("Video Identity cache rootが安全ではありません")
        entry_path = self._entry_path(logical_source_key)
        temporary_path = self._root / f".{entry_path.name}.{uuid4().hex}.tmp"
        payload = {
            "schema": _SCHEMA,
            "engine_version": self._engine_version,
            "logical_source_key": logical_source_key,
            "size_bytes": stat.st_size,
            "modified_at_ns": stat.st_mtime_ns,
            "fingerprint": fingerprint,
            "work_unit_fingerprint": _work_unit_fingerprint(
                self._engine_version,
                logical_source_key,
                stat,
            ),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            with temporary_path.open("x", encoding="utf-8") as temporary_file:
                json.dump(
                    payload,
                    temporary_file,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                temporary_file.write("\n")
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, entry_path)
            _fsync_directory(self._root)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _entry_path(self, logical_source_key: str) -> Path:
        return self._root / f"{logical_source_key}.json"

    def _root_is_safe(self) -> bool:
        return not self._root.is_symlink() and (
            not self._root.exists() or self._root.is_dir()
        )

    def _prepare_root(self) -> None:
        """長時間hashの前にdurable rootが安全かつ書込可能か検証する。"""
        if self._root_prepared:
            return
        if not self._root_is_safe():
            raise RuntimeError("Video Identity cache rootが安全ではありません")
        self._root.mkdir(parents=True, exist_ok=True)
        if not self._root_is_safe():
            raise RuntimeError("Video Identity cache rootが安全ではありません")
        probe_path = self._root / f".write-probe-{uuid4().hex}"
        try:
            with probe_path.open("x+b") as probe:
                probe.flush()
                os.fsync(probe.fileno())
        finally:
            probe_path.unlink(missing_ok=True)
        _fsync_directory(self._root)
        self._root_prepared = True

    def _observe_resolution(self, fingerprint: str, *, reused: bool) -> None:
        """SHA計算の確定量をpathなしのcheckpoint eventとして通知する。"""
        if self._observer is None:
            return
        self._observer.observe(
            ProgressEvent(
                kind="cache",
                severity="info",
                work_unit_fingerprint=fingerprint,
                cache_hit_count=1 if reused else 0,
                cache_miss_count=0 if reused else 1,
                reuse_count=1 if reused else 0,
                recompute_count=0 if reused else 1,
                work_unit_kind=CheckpointOperation.VIDEO_IDENTITY.value,
                reason_code="video-identity-checkpoint",
            )
        )

    @staticmethod
    def _remove_invalid_entry(entry_path: Path) -> None:
        """認識位置にある不正entryをcacheとして安全に削除する。"""
        try:
            if entry_path.is_symlink() or entry_path.is_file():
                entry_path.unlink()
            elif entry_path.is_dir():
                shutil.rmtree(entry_path)
        except OSError:
            return


def _logical_source_key(input_folder: Path, video_path: Path) -> str:
    try:
        relative_path = video_path.relative_to(input_folder).as_posix()
    except ValueError:
        msg = "Video SourceはVideo Input Folder配下である必要があります"
        raise ValueError(msg) from None
    input_identity = input_folder.absolute().as_posix()
    payload = json.dumps(
        {
            "input_folder": input_identity,
            "relative_path": relative_path,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(
        b"game-screen-pick/logical-video-source@1\0" + payload
    ).hexdigest()


def _validate_stable_snapshot(
    before: os.stat_result,
    after: os.stat_result,
) -> None:
    if source_snapshot_signature(before) != source_snapshot_signature(after):
        msg = "Video Set snapshotがfingerprint計算中に変更されました"
        raise ValueError(msg)


def _work_unit_fingerprint(
    engine_version: str,
    logical_source_key: str,
    stat: os.stat_result,
) -> str:
    payload = json.dumps(
        {
            "engine_version": engine_version,
            "logical_source_key": logical_source_key,
            "size_bytes": stat.st_size,
            "modified_at_ns": stat.st_mtime_ns,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(
        b"game-screen-pick/video-identity-work-unit@1\0" + payload
    ).hexdigest()


def _fsync_directory(directory: Path) -> None:
    """Identity entryのdirectory renameを永続化し、失敗を成功扱いにしない。"""
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_timezone_aware_iso_datetime(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.tzinfo is not None
