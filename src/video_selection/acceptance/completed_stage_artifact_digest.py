"""Completed Stageの実artifact内容をcanonical digestへまとめる。"""

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import cast

from ..models.completed_stage import CompletedStage

_CACHE_NAMESPACES = ("videos", "video-sets")
_VOLATILE_PERFORMANCE_KEYS = frozenset(
    {
        "cpu_seconds",
        "duration_seconds",
        "input_seconds_per_wall_second",
        "wall_seconds",
    }
)


def completed_stage_artifact_digest(
    cache_folder: Path,
    completed_stages: tuple[CompletedStage, ...],
) -> str:
    """manifestで検証した実artifactの意味的内容を一つのdigestで返す。"""
    if not completed_stages:
        raise ValueError("Completed Stage artifactがありません")
    stages = [
        _canonical_stage_artifacts(
            _locate_stage_folder(cache_folder, completed),
            completed,
        )
        for completed in completed_stages
    ]
    canonical = json.dumps(
        stages,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _locate_stage_folder(
    cache_folder: Path,
    completed: CompletedStage,
) -> Path:
    matches: list[Path] = []
    for namespace in _CACHE_NAMESPACES:
        namespace_root = cache_folder / namespace
        if not namespace_root.is_dir() or namespace_root.is_symlink():
            continue
        for subject_root in namespace_root.iterdir():
            if not subject_root.is_dir() or subject_root.is_symlink():
                continue
            stage_folder = (
                subject_root / completed.stage.value / completed.fingerprint.value
            )
            if stage_folder.is_dir() and not stage_folder.is_symlink():
                matches.append(stage_folder)
    if len(matches) != 1:
        raise ValueError("Completed Stage artifactの保存先を一意に解決できません")
    return matches[0]


def _canonical_stage_artifacts(
    stage_folder: Path,
    completed: CompletedStage,
) -> dict[str, object]:
    manifest_path = stage_folder / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("Completed Stage manifestが不正です")
    manifest = _read_json_mapping(manifest_path, "Completed Stage manifest")
    if (
        manifest.get("schema") != "game-screen-pick/completed-stage@1.0.0"
        or manifest.get("status") != "completed"
        or manifest.get("stage") != completed.stage.value
        or manifest.get("stage_fingerprint") != completed.fingerprint.value
    ):
        raise ValueError("Completed Stage manifest identityが不正です")
    artifact_records = manifest.get("artifacts")
    if not isinstance(artifact_records, list):
        raise ValueError("Completed Stage manifest artifactsが不正です")
    canonical_artifacts = [
        _canonical_artifact_record(stage_folder, value) for value in artifact_records
    ]
    canonical_artifacts.sort(key=lambda value: cast(str, value["path"]))
    return {
        "stage": completed.stage.value,
        "stage_fingerprint": completed.fingerprint.value,
        "artifacts": canonical_artifacts,
    }


def _canonical_artifact_record(
    stage_folder: Path,
    value: object,
) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError("Completed Stage artifact recordが不正です")
    record = cast(dict[str, object], value)
    relative_path = record.get("path")
    expected_size = record.get("size_bytes")
    expected_digest = record.get("sha256")
    if (
        not isinstance(relative_path, str)
        or not isinstance(expected_size, int)
        or isinstance(expected_size, bool)
        or expected_size < 0
        or not _is_digest(expected_digest)
    ):
        raise ValueError("Completed Stage artifact recordが不正です")
    artifact_path = _safe_artifact_path(stage_folder, relative_path)
    try:
        content = artifact_path.read_bytes()
    except OSError:
        raise ValueError("Completed Stage artifactを読み込めません") from None
    if (
        len(content) != expected_size
        or hashlib.sha256(content).hexdigest() != expected_digest
    ):
        raise ValueError("Completed Stage artifact integrityが不正です")
    semantic_content = (
        _canonical_json_artifact(content)
        if relative_path == "artifact.json"
        else content
    )
    return {
        "path": relative_path,
        "sha256": hashlib.sha256(semantic_content).hexdigest(),
    }


def _safe_artifact_path(stage_folder: Path, relative_path: str) -> Path:
    pure_path = PurePosixPath(relative_path)
    if (
        pure_path.is_absolute()
        or not pure_path.parts
        or ".." in pure_path.parts
        or "." in pure_path.parts
    ):
        raise ValueError("Completed Stage artifact pathが不正です")
    path = stage_folder
    for part in pure_path.parts:
        path /= part
        if path.is_symlink():
            raise ValueError("Completed Stage artifact pathが不正です")
    if not path.is_file():
        raise ValueError("Completed Stage artifactがありません")
    return path


def _canonical_json_artifact(content: bytes) -> bytes:
    try:
        value: object = json.loads(content)
    except (TypeError, ValueError, UnicodeDecodeError):
        raise ValueError("Completed Stage JSON artifactが不正です") from None
    normalized = canonicalize_completed_stage_artifact_value(value)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def canonicalize_completed_stage_artifact_value(value: object) -> object:
    """run性能値だけを除きCompleted Stage artifactの意味を返す。"""
    if isinstance(value, dict):
        return {
            key: canonicalize_completed_stage_artifact_value(item)
            for key, item in value.items()
            if key not in _VOLATILE_PERFORMANCE_KEYS
        }
    if isinstance(value, list):
        return [canonicalize_completed_stage_artifact_value(item) for item in value]
    return value


def _read_json_mapping(path: Path, label: str) -> dict[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        raise ValueError(f"{label}を読み込めません") from None
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label}がobjectではありません")
    return cast(dict[str, object], value)


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
