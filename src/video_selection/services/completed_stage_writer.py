"""Completed Stage artifactとmanifestを確定する。"""

import hashlib
import json
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
        stage_folder = self._root / stage.value / fingerprint.value
        stage_folder.mkdir(parents=True, exist_ok=True)
        artifact_bytes = (
            json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode()
        artifact_path = stage_folder / "artifact.json"
        self._replace_bytes(artifact_path, artifact_bytes)
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
        self._replace_bytes(stage_folder / "manifest.json", manifest_bytes)
        return CompletedStage(stage=stage, fingerprint=fingerprint)

    @staticmethod
    def _replace_bytes(path: Path, content: bytes) -> None:
        """同じdirectory内のtemporary fileから一度で置換する。"""
        temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            temporary_path.write_bytes(content)
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)
