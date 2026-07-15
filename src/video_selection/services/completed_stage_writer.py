"""Completed Stage artifactとmanifestを確定する。"""

import fcntl
import hashlib
import json
import shutil
from pathlib import Path
from typing import cast
from uuid import uuid4

from ..models.completed_stage import CompletedStage
from ..models.processing_stage import ProcessingStage
from ..models.stage_fingerprint import StageFingerprint


class CompletedStageWriter:
    """artifactの後にcompletion manifestをatomicに保存する。"""

    def __init__(self, cache_folder: Path) -> None:
        self._root = cache_folder / "walking-skeleton"

    def write(
        self,
        stage: ProcessingStage,
        fingerprint: StageFingerprint,
        upstream_fingerprints: tuple[StageFingerprint, ...],
        artifact: dict[str, object],
    ) -> CompletedStage:
        """Stage artifactと完了manifestを保存する。"""
        stage_root = self._root / stage.value
        stage_root.mkdir(parents=True, exist_ok=True)
        lock_root = self._root / ".locks" / stage.value
        lock_root.mkdir(parents=True, exist_ok=True)
        lock_path = lock_root / f"{fingerprint.value}.lock"
        with lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                return self._write_locked(
                    stage,
                    fingerprint,
                    upstream_fingerprints,
                    artifact,
                    stage_root,
                )
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _write_locked(
        self,
        stage: ProcessingStage,
        fingerprint: StageFingerprint,
        upstream_fingerprints: tuple[StageFingerprint, ...],
        artifact: dict[str, object],
        stage_root: Path,
    ) -> CompletedStage:
        """fingerprint lockの内側でStageを一度だけ確定する。"""
        stage_folder = stage_root / fingerprint.value
        if (
            self.read(
                stage,
                fingerprint,
                upstream_fingerprints,
            )
            is not None
        ):
            return CompletedStage(stage=stage, fingerprint=fingerprint)

        temporary_folder = stage_root / f".{fingerprint.value}.{uuid4().hex}.tmp"
        temporary_folder.mkdir()
        try:
            artifact_bytes = (
                json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n"
            ).encode()
            (temporary_folder / "artifact.json").write_bytes(artifact_bytes)
            manifest = {
                "schema": "game-screen-pick/completed-stage@0",
                "status": "completed",
                "stage": stage.value,
                "fingerprint": fingerprint.value,
                "upstream_fingerprints": [item.value for item in upstream_fingerprints],
                "artifact": "artifact.json",
                "artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
            }
            manifest_bytes = (
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n"
            ).encode()
            (temporary_folder / "manifest.json").write_bytes(manifest_bytes)
            self._remove_partial_stage(stage_folder)
            temporary_folder.replace(stage_folder)
        finally:
            shutil.rmtree(temporary_folder, ignore_errors=True)
        return CompletedStage(stage=stage, fingerprint=fingerprint)

    def read(
        self,
        stage: ProcessingStage,
        fingerprint: StageFingerprint,
        upstream_fingerprints: tuple[StageFingerprint, ...],
    ) -> dict[str, object] | None:
        """検証済みCompleted Stage artifactを返す。"""
        stage_folder = self._root / stage.value / fingerprint.value
        try:
            artifact_bytes = (stage_folder / "artifact.json").read_bytes()
            artifact_value: object = json.loads(artifact_bytes)
            manifest: object = json.loads(
                (stage_folder / "manifest.json").read_text(encoding="utf-8")
            )
        except (OSError, TypeError, ValueError):
            return None
        if manifest != {
            "schema": "game-screen-pick/completed-stage@0",
            "status": "completed",
            "stage": stage.value,
            "fingerprint": fingerprint.value,
            "upstream_fingerprints": [item.value for item in upstream_fingerprints],
            "artifact": "artifact.json",
            "artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
        }:
            return None
        if not isinstance(artifact_value, dict) or not all(
            isinstance(key, str) for key in artifact_value
        ):
            return None
        return cast(dict[str, object], artifact_value)

    @staticmethod
    def _remove_partial_stage(stage_folder: Path) -> None:
        """同じfingerprint位置にあるpartial entryだけを取り除く。"""
        if stage_folder.is_symlink() or stage_folder.is_file():
            stage_folder.unlink()
        elif stage_folder.exists():
            shutil.rmtree(stage_folder)
