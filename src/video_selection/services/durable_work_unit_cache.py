"""長時間処理の最小Work Unitをatomicに確定する。"""

import fcntl
import hashlib
import json
import os
import shutil
import stat
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import cast
from uuid import uuid4

from ..models.checkpoint_operation import CheckpointOperation
from ..models.durable_work_unit_bundle import DurableWorkUnitBundle
from ..models.progress_event import ProgressEvent
from ..protocols.run_observer import RunObserver
from .checkpoint_version import checkpoint_version

WorkUnitArtifactProducer = Callable[[Path], dict[str, object]]
WorkUnitFaultInjector = Callable[[str], None]
WorkUnitBundleValidator = Callable[[DurableWorkUnitBundle], object]

_SCHEMA = "game-screen-pick/durable-work-unit@1.0.0"


class DurableWorkUnitCache:
    """一つの処理種別に属する独立Work Unitを再利用可能にする。"""

    def __init__(
        self,
        cache_folder: Path,
        *,
        subject_fingerprint: str,
        operation: CheckpointOperation,
        fault_injector: WorkUnitFaultInjector | None = None,
        observer: RunObserver | None = None,
    ) -> None:
        if not _is_sha256(subject_fingerprint):
            msg = "Work Unit subjectには64桁のSHA-256が必要です"
            raise ValueError(msg)
        self._cache_folder = cache_folder
        self._subject_fingerprint = subject_fingerprint
        self._operation = operation.value
        self._engine_version = checkpoint_version(operation)
        self._fault_injector = fault_injector or _ignore_fault_checkpoint
        self._observer = observer
        self._root = cache_folder / "work-units" / subject_fingerprint / operation.value

    def resolve(
        self,
        work_unit_key: str,
        semantic_input: Mapping[str, object],
        produce_artifacts: WorkUnitArtifactProducer,
        *,
        validate_bundle: WorkUnitBundleValidator | None = None,
    ) -> tuple[DurableWorkUnitBundle, bool]:
        """検証済みcheckpointを返し、なければ一度だけ生成して確定する。"""
        normalized_input = _normalize_json_mapping(semantic_input)
        fingerprint = self._fingerprint(work_unit_key, normalized_input)
        self._validate_root()
        self._root.mkdir(parents=True, exist_ok=True)
        lock_root = (
            self._cache_folder
            / ".locks"
            / "work-units"
            / self._subject_fingerprint
            / self._operation
        )
        lock_root.mkdir(parents=True, exist_ok=True)
        lock_path = lock_root / f"{fingerprint}.lock"
        with lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                cached = self._read(
                    fingerprint,
                    work_unit_key,
                    normalized_input,
                )
                if cached is not None and validate_bundle is not None:
                    try:
                        validate_bundle(cached)
                    except (
                        FileNotFoundError,
                        IsADirectoryError,
                        NotADirectoryError,
                        TypeError,
                        ValueError,
                    ):
                        _remove_partial_entry(self._root / fingerprint)
                        _fsync_directory(self._root)
                        cached = None
                if cached is not None:
                    self._observe_resolution(fingerprint, reused=True)
                    return cached, True
                self._write_locked(
                    fingerprint,
                    work_unit_key,
                    normalized_input,
                    produce_artifacts,
                )
                committed = self._read(
                    fingerprint,
                    work_unit_key,
                    normalized_input,
                )
                if committed is None:
                    msg = "確定直後のWork Unit checkpointを検証できませんでした"
                    raise RuntimeError(msg)
                if validate_bundle is not None:
                    try:
                        validate_bundle(committed)
                    except (
                        FileNotFoundError,
                        IsADirectoryError,
                        NotADirectoryError,
                        TypeError,
                        ValueError,
                    ):
                        _remove_partial_entry(self._root / fingerprint)
                        _fsync_directory(self._root)
                        raise
                self._observe_resolution(fingerprint, reused=False)
                return committed, False
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def discard(
        self,
        work_unit_key: str,
        semantic_input: Mapping[str, object],
    ) -> None:
        """認識済みfingerprintのcheckpointだけをlock内で削除する。"""
        normalized_input = _normalize_json_mapping(semantic_input)
        fingerprint = self._fingerprint(work_unit_key, normalized_input)
        self._validate_root()
        if not self._root.exists():
            return
        lock_root = (
            self._cache_folder
            / ".locks"
            / "work-units"
            / self._subject_fingerprint
            / self._operation
        )
        lock_root.mkdir(parents=True, exist_ok=True)
        lock_path = lock_root / f"{fingerprint}.lock"
        with lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                _remove_partial_entry(self._root / fingerprint)
                _fsync_directory(self._root)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def read(
        self,
        work_unit_key: str,
        semantic_input: Mapping[str, object],
        *,
        validate_bundle: WorkUnitBundleValidator | None = None,
        observe_reuse: bool = False,
    ) -> DurableWorkUnitBundle | None:
        """検証済みcheckpointを読み、要求時だけ再利用を通知する。"""
        normalized_input = _normalize_json_mapping(semantic_input)
        fingerprint = self._fingerprint(work_unit_key, normalized_input)
        self._validate_root()
        bundle = self._read(
            fingerprint,
            work_unit_key,
            normalized_input,
        )
        if bundle is None:
            return None
        if validate_bundle is not None:
            validate_bundle(bundle)
        if observe_reuse:
            self._observe_resolution(fingerprint, reused=True)
        return bundle

    def _write_locked(
        self,
        fingerprint: str,
        work_unit_key: str,
        semantic_input: dict[str, object],
        produce_artifacts: WorkUnitArtifactProducer,
    ) -> None:
        """fingerprint lock内でcheckpointをatomicに確定する。"""
        checkpoint_folder = self._root / fingerprint
        _remove_recognized_temporary_entries(self._root, fingerprint)
        temporary_folder = self._root / f".{fingerprint}.{uuid4().hex}.tmp"
        temporary_folder.mkdir()
        try:
            artifact = produce_artifacts(temporary_folder)
            if not isinstance(artifact, dict) or not all(
                isinstance(key, str) for key in artifact
            ):
                msg = "Work Unit producerはJSON objectを返す必要があります"
                raise TypeError(msg)
            if (temporary_folder / "artifact.json").exists() or (
                temporary_folder / "manifest.json"
            ).exists():
                msg = "producerは予約済みartifact名を作成できません"
                raise ValueError(msg)
            artifact_bytes = _json_bytes(artifact)
            (temporary_folder / "artifact.json").write_bytes(artifact_bytes)
            self._fault_injector("before-manifest")
            records = _artifact_records(temporary_folder)
            manifest = {
                "schema": _SCHEMA,
                "status": "completed",
                "operation": self._operation,
                "engine_version": self._engine_version,
                "work_unit_key": work_unit_key,
                "work_unit_fingerprint": fingerprint,
                "subject_fingerprint": self._subject_fingerprint,
                "semantic_input": semantic_input,
                "artifacts": records,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            (temporary_folder / "manifest.json").write_bytes(_json_bytes(manifest))
            self._fault_injector("after-manifest")
            _fsync_tree(temporary_folder)
            self._fault_injector("before-rename")
            _remove_partial_entry(checkpoint_folder)
            temporary_folder.replace(checkpoint_folder)
            _fsync_directory(self._root)
            self._fault_injector("after-rename")
        finally:
            shutil.rmtree(temporary_folder, ignore_errors=True)

    def _read(
        self,
        fingerprint: str,
        work_unit_key: str,
        semantic_input: dict[str, object],
    ) -> DurableWorkUnitBundle | None:
        """manifestと全artifactが一致するcheckpointだけを返す。"""
        checkpoint_folder = self._root / fingerprint
        artifact_path = checkpoint_folder / "artifact.json"
        manifest_path = checkpoint_folder / "manifest.json"
        if (
            checkpoint_folder.is_symlink()
            or artifact_path.is_symlink()
            or manifest_path.is_symlink()
        ):
            return None
        try:
            artifact_value: object = json.loads(artifact_path.read_bytes())
            manifest_value: object = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            records = _artifact_records(checkpoint_folder)
        except (
            FileNotFoundError,
            IsADirectoryError,
            NotADirectoryError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ):
            return None
        if (
            not isinstance(artifact_value, dict)
            or not all(isinstance(key, str) for key in artifact_value)
            or not isinstance(manifest_value, dict)
            or not all(isinstance(key, str) for key in manifest_value)
        ):
            return None
        manifest = cast(dict[str, object], manifest_value)
        completed_at = manifest.get("completed_at")
        if not _is_timezone_aware_iso_datetime(completed_at):
            return None
        expected = {
            "schema": _SCHEMA,
            "status": "completed",
            "operation": self._operation,
            "engine_version": self._engine_version,
            "work_unit_key": work_unit_key,
            "work_unit_fingerprint": fingerprint,
            "subject_fingerprint": self._subject_fingerprint,
            "semantic_input": semantic_input,
            "artifacts": records,
            "completed_at": completed_at,
        }
        if manifest != expected:
            return None
        return DurableWorkUnitBundle(
            artifact=cast(dict[str, object], artifact_value),
            root=checkpoint_folder,
        )

    def _fingerprint(
        self,
        work_unit_key: str,
        semantic_input: dict[str, object],
    ) -> str:
        if not work_unit_key:
            msg = "Work Unit keyは空にできません"
            raise ValueError(msg)
        return hashlib.sha256(
            json.dumps(
                {
                    "engine_version": self._engine_version,
                    "work_unit_key": work_unit_key,
                    "semantic_input": semantic_input,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

    def _observe_resolution(self, fingerprint: str, *, reused: bool) -> None:
        """semantic処理へ影響しないprivacy-safeなwork量を通知する。"""
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
                work_unit_kind=self._operation,
                reason_code=f"{self._operation}-checkpoint",
            )
        )

    def _validate_root(self) -> None:
        """既存rootのsymbolic link経由でcheckpointを書かない。"""
        for parts in (
            ("work-units", self._subject_fingerprint, self._operation),
            (".locks", "work-units", self._subject_fingerprint, self._operation),
        ):
            current = self._cache_folder
            if current.is_symlink():
                raise OSError("Work Unit cache rootにsymbolic linkは使えません")
            for part in parts:
                current /= part
                if current.is_symlink():
                    raise OSError("Work Unit cache rootにsymbolic linkは使えません")


def _normalize_json_mapping(value: Mapping[str, object]) -> dict[str, object]:
    normalized: object = json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    if not isinstance(normalized, dict) or not all(
        isinstance(key, str) for key in normalized
    ):
        msg = "Work Unit semantic inputにはJSON objectが必要です"
        raise TypeError(msg)
    return cast(dict[str, object], normalized)


def _artifact_records(folder: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(folder.rglob("*")):
        if path.name == "manifest.json" and path.parent == folder:
            continue
        try:
            mode = path.lstat().st_mode
        except (FileNotFoundError, NotADirectoryError):
            raise ValueError("Work Unit artifactが処理中に消失しました") from None
        if stat.S_ISLNK(mode):
            msg = "Work Unit artifactにsymbolic linkは使えません"
            raise ValueError(msg)
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            msg = "Work Unit artifactは通常fileである必要があります"
            raise ValueError(msg)
        content = path.read_bytes()
        records.append(
            {
                "path": path.relative_to(folder).as_posix(),
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    return records


def _remove_recognized_temporary_entries(root: Path, fingerprint: str) -> None:
    prefix = f".{fingerprint}."
    suffix = ".tmp"
    for path in root.iterdir():
        name = path.name
        if not name.startswith(prefix) or not name.endswith(suffix):
            continue
        token = name[len(prefix) : -len(suffix)]
        if len(token) != 32 or any(
            character not in "0123456789abcdef" for character in token
        ):
            continue
        _remove_partial_entry(path)


def _remove_partial_entry(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()


def _fsync_tree(folder: Path) -> None:
    for path in sorted(folder.rglob("*")):
        if path.is_file() and not path.is_symlink():
            with path.open("rb") as artifact_file:
                os.fsync(artifact_file.fileno())
    for path in sorted(
        (item for item in folder.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        _fsync_directory(path)
    _fsync_directory(folder)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _is_timezone_aware_iso_datetime(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _ignore_fault_checkpoint(_checkpoint: str) -> None:
    return
