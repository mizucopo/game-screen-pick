"""Completed Stage artifactとmanifestを確定する。"""

import fcntl
import hashlib
import json
import shutil
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, cast
from uuid import uuid4

from ..models.completed_stage import CompletedStage
from ..models.completed_stage_bundle import CompletedStageBundle
from ..models.processing_stage import ProcessingStage
from ..models.stage_fingerprint import StageFingerprint
from .stage_version import stage_version

CacheNamespace = Literal["videos", "video-sets"]
FaultInjector = Callable[[str], None]
ArtifactProducer = Callable[[Path], dict[str, object]]


class CompletedStageWriter:
    """artifactの後にcompletion manifestをatomicに保存する。"""

    def __init__(
        self,
        cache_folder: Path,
        *,
        subject_namespace: CacheNamespace,
        subject_fingerprint: str,
        fault_injector: FaultInjector | None = None,
    ) -> None:
        self._cache_folder = cache_folder
        self._subject_namespace = subject_namespace
        self._subject_fingerprint = subject_fingerprint
        self._root = cache_folder / subject_namespace / subject_fingerprint
        self._fault_injector = fault_injector or _ignore_fault_checkpoint
        if len(subject_fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in subject_fingerprint
        ):
            msg = "cache subject fingerprintには64桁のSHA-256が必要です"
            raise ValueError(msg)

    def write(
        self,
        stage: ProcessingStage,
        fingerprint: StageFingerprint,
        upstream_fingerprints: tuple[StageFingerprint, ...],
        semantic_input: Mapping[str, object],
        artifact: dict[str, object],
    ) -> CompletedStage:
        """Stage artifactと完了manifestを保存する。"""
        return self.write_artifacts(
            stage,
            fingerprint,
            upstream_fingerprints,
            semantic_input,
            lambda _stage_folder: artifact,
        )

    def write_artifacts(
        self,
        stage: ProcessingStage,
        fingerprint: StageFingerprint,
        upstream_fingerprints: tuple[StageFingerprint, ...],
        semantic_input: Mapping[str, object],
        produce_artifacts: ArtifactProducer,
    ) -> CompletedStage:
        """複数artifactを生成して完了manifestとatomicに保存する。"""
        stage_root = self._root / stage.value
        stage_root.mkdir(parents=True, exist_ok=True)
        lock_root = (
            self._cache_folder
            / ".locks"
            / self._subject_namespace
            / self._subject_fingerprint
            / stage.value
        )
        lock_root.mkdir(parents=True, exist_ok=True)
        lock_path = lock_root / f"{fingerprint.value}.lock"
        with lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                return self._write_locked(
                    stage,
                    fingerprint,
                    upstream_fingerprints,
                    semantic_input,
                    produce_artifacts,
                    stage_root,
                )
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _write_locked(
        self,
        stage: ProcessingStage,
        fingerprint: StageFingerprint,
        upstream_fingerprints: tuple[StageFingerprint, ...],
        semantic_input: Mapping[str, object],
        produce_artifacts: ArtifactProducer,
        stage_root: Path,
    ) -> CompletedStage:
        """fingerprint lockの内側でStageを一度だけ確定する。"""
        stage_folder = stage_root / fingerprint.value
        _remove_recognized_temporary_stages(stage_root, fingerprint)
        if (
            self.read_bundle(
                stage,
                fingerprint,
                upstream_fingerprints,
                semantic_input,
            )
            is not None
        ):
            return CompletedStage(
                stage=stage,
                fingerprint=fingerprint,
                upstream_fingerprints=upstream_fingerprints,
                semantic_input=_normalize_json_mapping(semantic_input),
            )

        temporary_folder = stage_root / f".{fingerprint.value}.{uuid4().hex}.tmp"
        temporary_folder.mkdir()
        try:
            artifact = produce_artifacts(temporary_folder)
            if not isinstance(artifact, dict) or not all(
                isinstance(key, str) for key in artifact
            ):
                msg = "Completed Stage producerはJSON objectを返す必要があります"
                raise TypeError(msg)
            if (temporary_folder / "artifact.json").exists() or (
                temporary_folder / "manifest.json"
            ).exists():
                msg = "producerは予約済みartifact名を作成できません"
                raise ValueError(msg)
            artifact_bytes = (
                json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n"
            ).encode()
            (temporary_folder / "artifact.json").write_bytes(artifact_bytes)
            self._fault_injector("before-manifest")
            artifact_records = _artifact_records(temporary_folder)
            manifest = {
                "schema": "game-screen-pick/completed-stage@1.0.0",
                "status": "completed",
                "stage": stage.value,
                "stage_version": stage_version(stage),
                "stage_fingerprint": fingerprint.value,
                "subject": {
                    "namespace": self._subject_namespace,
                    "fingerprint": self._subject_fingerprint,
                },
                "upstream_stage_fingerprints": [
                    item.value for item in upstream_fingerprints
                ],
                "semantic_input": _normalize_json_mapping(semantic_input),
                "artifacts": artifact_records,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            manifest_bytes = (
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n"
            ).encode()
            (temporary_folder / "manifest.json").write_bytes(manifest_bytes)
            self._fault_injector("after-manifest")
            self._fault_injector("before-rename")
            self._remove_partial_stage(stage_folder)
            temporary_folder.replace(stage_folder)
            self._fault_injector("after-rename")
        finally:
            shutil.rmtree(temporary_folder, ignore_errors=True)
        return CompletedStage(
            stage=stage,
            fingerprint=fingerprint,
            upstream_fingerprints=upstream_fingerprints,
            semantic_input=_normalize_json_mapping(semantic_input),
        )

    def read(
        self,
        stage: ProcessingStage,
        fingerprint: StageFingerprint,
        upstream_fingerprints: tuple[StageFingerprint, ...],
        semantic_input: Mapping[str, object],
    ) -> dict[str, object] | None:
        """検証済みCompleted Stage artifactを返す。"""
        bundle = self.read_bundle(
            stage,
            fingerprint,
            upstream_fingerprints,
            semantic_input,
        )
        return None if bundle is None else bundle.artifact

    def read_bundle(
        self,
        stage: ProcessingStage,
        fingerprint: StageFingerprint,
        upstream_fingerprints: tuple[StageFingerprint, ...],
        semantic_input: Mapping[str, object],
    ) -> CompletedStageBundle | None:
        """検証済みJSON artifactとStage rootを返す。"""
        stage_folder = self._root / stage.value / fingerprint.value
        artifact_path = stage_folder / "artifact.json"
        manifest_path = stage_folder / "manifest.json"
        if (
            stage_folder.is_symlink()
            or artifact_path.is_symlink()
            or manifest_path.is_symlink()
        ):
            return None
        try:
            artifact_bytes = artifact_path.read_bytes()
            artifact_value: object = json.loads(artifact_bytes)
            manifest_value: object = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, TypeError, ValueError):
            return None
        if not isinstance(manifest_value, dict) or not all(
            isinstance(key, str) for key in manifest_value
        ):
            return None
        manifest = cast(dict[str, object], manifest_value)
        completed_at = manifest.get("completed_at")
        if not _is_timezone_aware_iso_datetime(completed_at):
            return None
        try:
            artifact_records = _artifact_records(stage_folder)
        except OSError:
            return None
        expected_manifest = {
            "schema": "game-screen-pick/completed-stage@1.0.0",
            "status": "completed",
            "stage": stage.value,
            "stage_version": stage_version(stage),
            "stage_fingerprint": fingerprint.value,
            "subject": {
                "namespace": self._subject_namespace,
                "fingerprint": self._subject_fingerprint,
            },
            "upstream_stage_fingerprints": [
                item.value for item in upstream_fingerprints
            ],
            "semantic_input": _normalize_json_mapping(semantic_input),
            "artifacts": artifact_records,
            "completed_at": completed_at,
        }
        if manifest != expected_manifest:
            return None
        if not isinstance(artifact_value, dict) or not all(
            isinstance(key, str) for key in artifact_value
        ):
            return None
        return CompletedStageBundle(
            artifact=cast(dict[str, object], artifact_value),
            root=stage_folder,
        )

    @staticmethod
    def _remove_partial_stage(stage_folder: Path) -> None:
        """同じfingerprint位置にあるpartial entryだけを取り除く。"""
        if stage_folder.is_symlink() or stage_folder.is_file():
            stage_folder.unlink()
        elif stage_folder.exists():
            shutil.rmtree(stage_folder)


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
        msg = "semantic inputにはJSON objectが必要です"
        raise TypeError(msg)
    return cast(dict[str, object], normalized)


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


def _remove_recognized_temporary_stages(
    stage_root: Path,
    fingerprint: StageFingerprint,
) -> None:
    """同じfingerprintの正規形式temporary entryだけを削除する。"""
    prefix = f".{fingerprint.value}."
    suffix = ".tmp"
    for path in stage_root.iterdir():
        name = path.name
        if not name.startswith(prefix) or not name.endswith(suffix):
            continue
        token = name[len(prefix) : -len(suffix)]
        if len(token) != 32 or any(
            character not in "0123456789abcdef" for character in token
        ):
            continue
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def _artifact_records(stage_folder: Path) -> list[dict[str, object]]:
    """manifest以外の通常fileを相対path順でhash化する。"""
    records: list[dict[str, object]] = []
    for path in sorted(stage_folder.rglob("*")):
        if path.name == "manifest.json" and path.parent == stage_folder:
            continue
        if path.is_symlink():
            msg = "Completed Stage artifactにsymbolic linkは使えません"
            raise OSError(msg)
        if path.is_dir():
            continue
        if not path.is_file():
            msg = "Completed Stage artifactは通常fileである必要があります"
            raise OSError(msg)
        relative_path = path.relative_to(stage_folder).as_posix()
        content = path.read_bytes()
        records.append(
            {
                "path": relative_path,
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    return records
