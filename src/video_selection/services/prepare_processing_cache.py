"""Input Lock内で行うprocessing cache準備。"""

import json
import re
import shutil
from pathlib import Path
from typing import cast
from uuid import uuid4

from ..models.legacy_cache_cleanup_diagnostic import LegacyCacheCleanupDiagnostic
from ..models.processing_stage import ProcessingStage
from ..vision.vision_contract import CANDIDATE_ANNOTATION_STAGE_CONTRACT_VERSION
from .input_folder_lock import InputFolderLock

_CANDIDATE_ANNOTATION_CONTRACT_PATTERN = re.compile(
    r"candidate-annotation-stage-v([1-9][0-9]*)"
)


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
    legacy_video_identities = cache_folder / "video-identities"
    if legacy_video_identities.is_dir() and not legacy_video_identities.is_symlink():
        removed_bytes += _directory_content_bytes(legacy_video_identities)
        shutil.rmtree(legacy_video_identities)
        removed_entry_count += 1
    annotation_diagnostic = _remove_legacy_candidate_annotation_cache(cache_folder)
    removed_entry_count += annotation_diagnostic.removed_entry_count
    removed_bytes += annotation_diagnostic.removed_bytes
    return LegacyCacheCleanupDiagnostic(
        removed_entry_count=removed_entry_count,
        removed_bytes=removed_bytes,
    )


def _remove_legacy_candidate_annotation_cache(
    cache_folder: Path,
) -> LegacyCacheCleanupDiagnostic:
    """認識済み旧contractのCandidate Annotation artifactだけを削除する。"""
    current_revision = _candidate_annotation_contract_revision(
        CANDIDATE_ANNOTATION_STAGE_CONTRACT_VERSION
    )
    if current_revision is None:
        raise RuntimeError("現行Candidate Annotation Stage Contractが不正です")
    removed_entry_count = 0
    removed_bytes = 0
    video_sets_root = cache_folder / "video-sets"
    if not _is_sha256_directory(video_sets_root, require_sha256_name=False):
        return LegacyCacheCleanupDiagnostic(
            removed_entry_count=0,
            removed_bytes=0,
        )
    for subject_folder in sorted(video_sets_root.iterdir()):
        if not _is_sha256_directory(subject_folder):
            continue
        stage_root = subject_folder / ProcessingStage.ANNOTATE_CANDIDATE.value
        if not _is_sha256_directory(stage_root, require_sha256_name=False):
            continue
        for stage_folder in sorted(stage_root.iterdir()):
            if not _is_sha256_directory(stage_folder):
                continue
            revision = _recognized_candidate_annotation_contract_revision(
                stage_folder,
                subject_folder.name,
            )
            if revision is None or revision >= current_revision:
                continue
            removed_bytes += _directory_content_bytes(stage_folder)
            shutil.rmtree(stage_folder)
            removed_entry_count += 1
    return LegacyCacheCleanupDiagnostic(
        removed_entry_count=removed_entry_count,
        removed_bytes=removed_bytes,
    )


def _recognized_candidate_annotation_contract_revision(
    stage_folder: Path,
    subject_fingerprint: str,
) -> int | None:
    """認識済みCandidate Annotation manifestのcontract revisionを返す。"""
    manifest_path = stage_folder / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        return None
    try:
        value: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return None
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        return None
    manifest = cast(dict[str, object], value)
    subject = manifest.get("subject")
    semantic_input = manifest.get("semantic_input")
    if (
        manifest.get("schema") != "game-screen-pick/completed-stage@1.0.0"
        or manifest.get("status") != "completed"
        or manifest.get("stage") != ProcessingStage.ANNOTATE_CANDIDATE.value
        or manifest.get("stage_fingerprint") != stage_folder.name
        or subject
        != {
            "namespace": "video-sets",
            "fingerprint": subject_fingerprint,
        }
        or not isinstance(semantic_input, dict)
        or not all(isinstance(key, str) for key in semantic_input)
    ):
        return None
    return _candidate_annotation_contract_revision(
        semantic_input.get("stage_contract_version")
    )


def _candidate_annotation_contract_revision(value: object) -> int | None:
    """versioned Candidate Annotation contractの正のrevisionを返す。"""
    if not isinstance(value, str):
        return None
    matched = _CANDIDATE_ANNOTATION_CONTRACT_PATTERN.fullmatch(value)
    return None if matched is None else int(matched.group(1))


def _is_sha256_directory(
    path: Path,
    *,
    require_sha256_name: bool = True,
) -> bool:
    """symlinkでない通常directoryと、必要ならSHA-256名を検証する。"""
    if path.is_symlink() or not path.is_dir():
        return False
    return not require_sha256_name or (
        len(path.name) == 64
        and all(character in "0123456789abcdef" for character in path.name)
    )


def _directory_content_bytes(directory: Path) -> int:
    return sum(
        path.stat().st_size
        for path in directory.rglob("*")
        if path.is_file() and not path.is_symlink()
    )
