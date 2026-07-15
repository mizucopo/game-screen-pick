"""Completed Stage artifactとmanifestを確定する。"""

import hashlib
import json
import shutil
from pathlib import Path
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
        stage_folder = stage_root / fingerprint.value
        if self._is_completed(
            stage_folder,
            stage,
            fingerprint,
            upstream_fingerprints,
        ):
            return CompletedStage(stage=stage, fingerprint=fingerprint)

        temporary_folder = stage_root / f".{fingerprint.value}.{uuid4().hex}.tmp"
        temporary_folder.mkdir()
        artifact_bytes = (
            json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
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
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode()
        (temporary_folder / "manifest.json").write_bytes(manifest_bytes)
        try:
            self._remove_partial_stage(stage_folder)
            temporary_folder.replace(stage_folder)
        finally:
            shutil.rmtree(temporary_folder, ignore_errors=True)
        return CompletedStage(stage=stage, fingerprint=fingerprint)

    @staticmethod
    def _is_completed(
        stage_folder: Path,
        stage: ProcessingStage,
        fingerprint: StageFingerprint,
        upstream_fingerprints: tuple[StageFingerprint, ...],
    ) -> bool:
        """既存folderが再利用可能なCompleted Stageかを返す。"""
        try:
            artifact_bytes = (stage_folder / "artifact.json").read_bytes()
            json.loads(artifact_bytes)
            manifest: object = json.loads(
                (stage_folder / "manifest.json").read_text(encoding="utf-8")
            )
        except (OSError, TypeError, ValueError):
            return False
        return manifest == {
            "schema": "game-screen-pick/completed-stage@0",
            "status": "completed",
            "stage": stage.value,
            "fingerprint": fingerprint.value,
            "upstream_fingerprints": [item.value for item in upstream_fingerprints],
            "artifact": "artifact.json",
            "artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
        }

    @staticmethod
    def _remove_partial_stage(stage_folder: Path) -> None:
        """同じfingerprint位置にあるpartial entryだけを取り除く。"""
        if stage_folder.is_symlink() or stage_folder.is_file():
            stage_folder.unlink()
        elif stage_folder.exists():
            shutil.rmtree(stage_folder)
