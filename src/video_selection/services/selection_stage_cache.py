"""選定前fingerprintからSelect Images Completed Stageを索引する。"""

import json
from pathlib import Path
from typing import cast
from uuid import uuid4

from ..models.completed_stage import CompletedStage
from ..models.processing_stage import ProcessingStage
from ..models.stage_fingerprint import StageFingerprint
from .build_stage_fingerprint import build_stage_fingerprint
from .completed_stage_writer import CompletedStageWriter

_INDEX_SCHEMA = "game-screen-pick/selection-stage-index@1.0.0"


class SelectionStageCache:
    """選定を実行せず検証済み選定artifactへ到達するcache index。"""

    def __init__(
        self,
        cache_folder: Path,
        *,
        video_set_fingerprint: str,
    ) -> None:
        self._cache_folder = cache_folder
        self._video_set_fingerprint = video_set_fingerprint
        self._writer = CompletedStageWriter(
            cache_folder,
            subject_namespace="video-sets",
            subject_fingerprint=video_set_fingerprint,
        )
        self._stage_root = (
            cache_folder
            / "video-sets"
            / video_set_fingerprint
            / ProcessingStage.SELECT_IMAGES.value
        )
        self._index_root = (
            cache_folder
            / ".indexes"
            / "video-sets"
            / video_set_fingerprint
            / ProcessingStage.SELECT_IMAGES.value
        )

    def read(
        self,
        request_fingerprint: StageFingerprint,
    ) -> tuple[dict[str, object], CompletedStage] | None:
        """選定前fingerprintに対応する検証済みartifactとStageを復元する。"""
        indexed = self._read_index(request_fingerprint)
        if indexed is not None:
            return indexed
        recovered = self._recover_completed_selection(request_fingerprint)
        if recovered is None:
            return None
        self.record(request_fingerprint, recovered[1])
        return recovered

    def _read_index(
        self,
        request_fingerprint: StageFingerprint,
    ) -> tuple[dict[str, object], CompletedStage] | None:
        """request indexが指すCompleted Stageをintegrity検証する。"""
        path = self._index_root / f"{request_fingerprint.value}.json"
        if path.is_symlink():
            return None
        try:
            value: object = json.loads(path.read_text(encoding="utf-8"))
        except (
            FileNotFoundError,
            IsADirectoryError,
            NotADirectoryError,
            TypeError,
            ValueError,
        ):
            return None
        if not isinstance(value, dict) or not all(
            isinstance(key, str) for key in value
        ):
            return None
        index = cast(dict[str, object], value)
        stage_fingerprint_value = index.get("stage_fingerprint")
        raw_upstream = index.get("upstream_stage_fingerprints")
        semantic_input = index.get("semantic_input")
        if (
            index.get("schema") != _INDEX_SCHEMA
            or index.get("request_fingerprint") != request_fingerprint.value
            or not _is_fingerprint(stage_fingerprint_value)
            or not isinstance(raw_upstream, list)
            or not all(_is_fingerprint(item) for item in raw_upstream)
            or not isinstance(semantic_input, dict)
            or not all(isinstance(key, str) for key in semantic_input)
        ):
            return None
        upstream = tuple(StageFingerprint(cast(str, item)) for item in raw_upstream)
        typed_semantic_input = cast(dict[str, object], semantic_input)
        stage_fingerprint = StageFingerprint(cast(str, stage_fingerprint_value))
        if (
            typed_semantic_input.get("selection_request_fingerprint")
            != request_fingerprint.value
            or build_stage_fingerprint(
                ProcessingStage.SELECT_IMAGES,
                upstream,
                typed_semantic_input,
            )
            != stage_fingerprint
        ):
            return None
        artifact = self._writer.read(
            ProcessingStage.SELECT_IMAGES,
            stage_fingerprint,
            upstream,
            typed_semantic_input,
        )
        if artifact is None:
            return None
        return (
            artifact,
            CompletedStage(
                ProcessingStage.SELECT_IMAGES,
                stage_fingerprint,
                upstream,
                typed_semantic_input,
            ),
        )

    def _recover_completed_selection(
        self,
        request_fingerprint: StageFingerprint,
    ) -> tuple[dict[str, object], CompletedStage] | None:
        """欠落indexをsemantic inputが一致するCompleted Stageから回復する。"""
        if self._stage_root.is_symlink():
            return None
        try:
            stage_folders = tuple(self._stage_root.iterdir())
        except (FileNotFoundError, NotADirectoryError):
            return None
        matches: list[tuple[dict[str, object], CompletedStage]] = []
        for stage_folder in stage_folders:
            restored = self._read_completed_stage_folder(
                stage_folder,
                request_fingerprint,
            )
            if restored is not None:
                matches.append(restored)
        return matches[0] if len(matches) == 1 else None

    def _read_completed_stage_folder(
        self,
        stage_folder: Path,
        request_fingerprint: StageFingerprint,
    ) -> tuple[dict[str, object], CompletedStage] | None:
        """一つのStage folderをindex候補として完全検証する。"""
        if (
            stage_folder.is_symlink()
            or not stage_folder.is_dir()
            or not _is_fingerprint(stage_folder.name)
        ):
            return None
        manifest_path = stage_folder / "manifest.json"
        if manifest_path.is_symlink():
            return None
        try:
            value: object = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (
            FileNotFoundError,
            IsADirectoryError,
            NotADirectoryError,
            TypeError,
            ValueError,
        ):
            return None
        if not isinstance(value, dict) or not all(
            isinstance(key, str) for key in value
        ):
            return None
        manifest = cast(dict[str, object], value)
        raw_upstream = manifest.get("upstream_stage_fingerprints")
        semantic_input = manifest.get("semantic_input")
        if (
            not isinstance(raw_upstream, list)
            or not all(_is_fingerprint(item) for item in raw_upstream)
            or not isinstance(semantic_input, dict)
            or not all(isinstance(key, str) for key in semantic_input)
        ):
            return None
        typed_semantic_input = cast(dict[str, object], semantic_input)
        if (
            typed_semantic_input.get("selection_request_fingerprint")
            != request_fingerprint.value
        ):
            return None
        upstream = tuple(StageFingerprint(cast(str, item)) for item in raw_upstream)
        stage_fingerprint = StageFingerprint(stage_folder.name)
        if (
            build_stage_fingerprint(
                ProcessingStage.SELECT_IMAGES,
                upstream,
                typed_semantic_input,
            )
            != stage_fingerprint
        ):
            return None
        artifact = self._writer.read(
            ProcessingStage.SELECT_IMAGES,
            stage_fingerprint,
            upstream,
            typed_semantic_input,
        )
        if artifact is None:
            return None
        return (
            artifact,
            CompletedStage(
                ProcessingStage.SELECT_IMAGES,
                stage_fingerprint,
                upstream,
                typed_semantic_input,
            ),
        )

    def record(
        self,
        request_fingerprint: StageFingerprint,
        completed: CompletedStage,
    ) -> None:
        """確定済みSelect Images Stageへのindexをatomicに保存する。"""
        if completed.stage is not ProcessingStage.SELECT_IMAGES:
            raise ValueError("Selection cacheにはSelect Images Stageが必要です")
        if (
            completed.semantic_input.get("selection_request_fingerprint")
            != request_fingerprint.value
            or build_stage_fingerprint(
                ProcessingStage.SELECT_IMAGES,
                completed.upstream_fingerprints,
                completed.semantic_input,
            )
            != completed.fingerprint
        ):
            raise ValueError("Selection cache Stage identityが一致しません")
        self._index_root.mkdir(parents=True, exist_ok=True)
        path = self._index_root / f"{request_fingerprint.value}.json"
        temporary = self._index_root / (
            f".{request_fingerprint.value}.{uuid4().hex}.tmp"
        )
        value = {
            "schema": _INDEX_SCHEMA,
            "request_fingerprint": request_fingerprint.value,
            "stage_fingerprint": completed.fingerprint.value,
            "upstream_stage_fingerprints": [
                item.value for item in completed.upstream_fingerprints
            ],
            "semantic_input": dict(completed.semantic_input),
        }
        try:
            temporary.write_text(
                json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def discard(
        self,
        request_fingerprint: StageFingerprint,
        completed: CompletedStage,
    ) -> None:
        """不正な選定Stageと対応する認識済みindexだけを削除する。"""
        self._writer.discard(completed.stage, completed.fingerprint)
        index_path = self._index_root / f"{request_fingerprint.value}.json"
        if index_path.is_symlink() or index_path.is_file():
            index_path.unlink()
        elif index_path.exists():
            raise ValueError("Selection cache indexが通常fileではありません")


def _is_fingerprint(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
