"""Acceptance Runの一試行を実行し、完了成果物を検証する。"""

import hashlib
import json
import time
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import cast

from ..application.internal_run_controller import InternalRunController
from ..models.effective_configuration import EffectiveConfiguration
from ..models.processing_stage import ProcessingStage
from ..models.resolved_models import ResolvedModels
from ..models.run_failure import RunFailure
from ..models.run_outcome import RunOutcome
from ..models.stage_fingerprint import StageFingerprint
from ..services.build_stage_fingerprint import build_stage_fingerprint
from ..services.canonical_report_semantic_digest import (
    canonical_report_semantic_digest,
)
from ..services.completed_stage_writer import CompletedStageWriter
from ..services.progress_stream_observer import ProgressStreamObserver
from ..services.render_human_selection_report import render_human_selection_report
from ..services.run_progress_tracker import RunProgressTracker
from .acceptance_attempt_journal import AcceptanceAttemptJournal
from .acceptance_run_attempt_observer import AcceptanceRunAttemptObserver
from .build_real_application import build_real_application
from .completed_stage_artifact_digest import completed_stage_artifact_digest
from .disk_usage_monitor import DiskUsageMonitor
from .gpu_resource_monitor import GpuResourceMonitor

AcceptanceRunAttemptExecutionResult = tuple[
    int,
    dict[str, object],
    dict[str, object] | None,
    dict[str, object] | None,
]


def execute_acceptance_run_attempt(
    *,
    configuration: EffectiveConfiguration,
    resolved_models: ResolvedModels,
    suite_root: Path,
) -> AcceptanceRunAttemptExecutionResult:
    """model freeze後からatomic publicationまでの一試行を測定する。"""
    attempt_journal = AcceptanceAttemptJournal(
        suite_root / "work" / "active-attempt.json"
    )
    observer = AcceptanceRunAttemptObserver(
        ProgressStreamObserver(),
        snapshot_writer=(
            attempt_journal.record_snapshot if attempt_journal.exists else None
        ),
    )
    progress = RunProgressTracker(observer)
    application = build_real_application(
        configuration,
        resolved_models,
        observer,
        progress,
    )
    gpu_monitor = GpuResourceMonitor(
        ollama_host=configuration.ollama_host,
        stage_provider=lambda: observer.current_stage,
    )
    disk_monitor = DiskUsageMonitor(
        working_root=suite_root / "work",
        output_parent=suite_root / "outputs",
        cache_folder=configuration.processing_cache_folder,
    )
    started_at = time.monotonic()
    gpu_monitor.start()
    disk_monitor.start()
    try:
        exit_code, result = InternalRunController(progress).execute(
            lambda: application.run(configuration)
        )
    finally:
        run_completed_at = time.monotonic()
        disk_metrics = disk_monitor.stop()
        gpu_metrics = gpu_monitor.stop()
    duration_seconds = run_completed_at - started_at
    attempt_record: dict[str, object] = {
        "duration_seconds": duration_seconds,
        **observer.attempt_metrics(),
        **disk_metrics,
        **gpu_metrics,
        "resource_sampling_complete": (
            disk_metrics.get("disk_sampling_complete") is True
            and gpu_metrics.get("resource_sampling_complete") is True
        ),
    }
    application_parallelism = getattr(
        application,
        "video_scan_parallelism_diagnostics",
        {},
    )
    if isinstance(application_parallelism, dict) and application_parallelism:
        attempt_record["video_scan_parallelism"] = dict(application_parallelism)
    if exit_code != 0:
        if not isinstance(result, RunFailure):
            raise AssertionError
        attempt_record.update(
            {
                "operation_status": "failed",
                "failure_reason": result.reason_code,
                "failure_exit_code": result.exit_code,
            }
        )
        return int(exit_code), attempt_record, None, None
    if not isinstance(result, RunOutcome):
        raise AssertionError
    report_path = result.output_folder / "report.json"
    markdown_path = result.output_folder / "report.md"
    report = _read_json_object(report_path)
    report_parallelism = video_scan_parallelism_diagnostics(report)
    if (
        not result.reused_completed_publication
        and isinstance(application_parallelism, dict)
        and application_parallelism
        and application_parallelism != report_parallelism
    ):
        raise ValueError("Video Scan parallelism診断がpublicationと一致しません")
    selection_stage = next(
        stage
        for stage in reversed(result.completed_stages)
        if stage.stage is ProcessingStage.SELECT_IMAGES
    )
    attempt_record.update(
        {
            "operation_status": "completed",
            "selected_count": result.selected_count,
            "requested_count": result.requested_count,
            "canonical_report_sha256": _file_digest(report_path),
            "canonical_markdown_sha256": _file_digest(markdown_path),
            "normalized_result_digest": normalized_result_digest(report),
            "stage_artifact_content_digest": completed_stage_artifact_digest(
                configuration.processing_cache_folder,
                result.completed_stages,
            ),
            "video_scan_parallelism": report_parallelism,
            "selection_stage_fingerprint": selection_stage.fingerprint.value,
            "video_set": _video_set_record(report),
            "speech_runtime_identity": _speech_runtime_identity(report),
        }
    )
    selection_artifact = _selection_artifact(
        configuration.processing_cache_folder,
        attempt_record,
    )
    return 0, attempt_record, report, selection_artifact


def load_completed_run_evidence(
    *,
    configuration: EffectiveConfiguration,
    run_record: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """durable output/cacheからcompleted runのworksheet sourceを復元する。"""
    report = load_completed_run_report(
        configuration=configuration,
        run_record=run_record,
    )
    return report, _selection_artifact(
        configuration.processing_cache_folder,
        run_record,
    )


def load_completed_run_report(
    *,
    configuration: EffectiveConfiguration,
    run_record: Mapping[str, object],
) -> dict[str, object]:
    """durable canonical reportと公開画像がrun確定時のままか再検証する。"""
    output_folder = configuration.output_folder
    report_path = output_folder / "report.json"
    if (
        output_folder.is_symlink()
        or not output_folder.is_dir()
        or report_path.is_symlink()
        or not report_path.is_file()
    ):
        raise ValueError("Completed runのcanonical report artifactがありません")
    expected_report_digest = _fingerprint(
        run_record.get("canonical_report_sha256"),
        "canonical report digest",
    )
    if _file_digest(report_path) != expected_report_digest:
        raise ValueError("Completed runのcanonical report artifactが一致しません")
    report = _read_json_object(report_path)
    expected_digest = run_record.get("normalized_result_digest")
    if (
        not isinstance(expected_digest, str)
        or normalized_result_digest(report) != expected_digest
    ):
        raise ValueError("Completed runのcanonical report digestが一致しません")
    _validate_completed_markdown(report, output_folder, run_record)
    _validate_selected_output_artifacts(report, output_folder)
    return report


def public_run_record(value: Mapping[str, object]) -> dict[str, object]:
    """suite stateのrunからrun-level Video Set重複値を除いて返す。"""
    private_state_keys = {
        "canonical_markdown_sha256",
        "canonical_report_sha256",
        "video_set",
    }
    return {key: item for key, item in value.items() if key not in private_state_keys}


def _validate_completed_markdown(
    report: dict[str, object],
    output_folder: Path,
    run_record: Mapping[str, object],
) -> None:
    """完了時hashまたは決定的projectionから公開Markdownを再検証する。"""
    markdown_path = output_folder / "report.md"
    if markdown_path.is_symlink() or not markdown_path.is_file():
        raise ValueError("Completed runのMarkdown artifactがありません")
    expected_digest = run_record.get("canonical_markdown_sha256")
    if expected_digest is not None:
        if _file_digest(markdown_path) != _fingerprint(
            expected_digest,
            "canonical Markdown digest",
        ):
            raise ValueError("Completed runのMarkdown artifactが一致しません")
        return
    try:
        actual = markdown_path.read_text(encoding="utf-8")
        expected = render_human_selection_report(report)
    except (OSError, KeyError, TypeError, ValueError):
        raise ValueError("Completed runのMarkdown artifactを検証できません") from None
    if actual != expected:
        raise ValueError("Completed runのMarkdown artifactが一致しません")


def _selection_artifact(
    cache_folder: Path,
    run_record: Mapping[str, object],
) -> dict[str, object]:
    fingerprint = _stage_fingerprint(
        run_record.get("selection_stage_fingerprint"),
        "selection fingerprint",
    )
    video_set = _mapping(run_record.get("video_set"), "Video Set")
    subject_fingerprint = _fingerprint(
        video_set.get("fingerprint"),
        "Video Set fingerprint",
    )
    manifest_path = (
        cache_folder
        / "video-sets"
        / subject_fingerprint
        / ProcessingStage.SELECT_IMAGES.value
        / fingerprint.value
        / "manifest.json"
    )
    manifest = _read_json_object(manifest_path)
    semantic_input = _mapping(
        manifest.get("semantic_input"),
        "Selection Stage semantic input",
    )
    upstream_values = manifest.get("upstream_stage_fingerprints")
    if not isinstance(upstream_values, list):
        raise ValueError("Selection Stage upstream fingerprintsが不正です")
    upstream = tuple(
        _stage_fingerprint(value, "upstream fingerprint") for value in upstream_values
    )
    if (
        build_stage_fingerprint(
            ProcessingStage.SELECT_IMAGES,
            upstream,
            semantic_input,
        )
        != fingerprint
    ):
        raise ValueError("Selection Stage fingerprintがsemantic inputと一致しません")
    bundle = CompletedStageWriter(
        cache_folder,
        subject_namespace="video-sets",
        subject_fingerprint=subject_fingerprint,
    ).read_bundle(
        ProcessingStage.SELECT_IMAGES,
        fingerprint,
        upstream,
        semantic_input,
    )
    if bundle is None:
        raise ValueError("Selection Stage artifactのintegrity検証に失敗しました")
    return bundle.artifact


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        raise ValueError("Acceptance run artifactを読み込めません") from None
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError("Acceptance run artifactがobjectではありません")
    return cast(dict[str, object], value)


def _validate_selected_output_artifacts(
    report: Mapping[str, object],
    output_folder: Path,
) -> None:
    """selected outputのpath、byte数、SHA-256をreportと照合する。"""
    selected = report.get("selected")
    if not isinstance(selected, list):
        raise ValueError("Canonical reportのselectedが不正です")
    observed_paths: set[str] = set()
    for value in selected:
        if not isinstance(value, dict):
            raise ValueError("Canonical reportのselected recordが不正です")
        output = value.get("output")
        if not isinstance(output, dict):
            raise ValueError("Canonical reportのselected outputが不正です")
        relative_path = _selected_output_relative_path(output.get("relative_path"))
        if relative_path in observed_paths:
            raise ValueError("Canonical reportのselected output pathが重複しています")
        observed_paths.add(relative_path)
        artifact_path = _selected_output_artifact_path(
            output_folder,
            relative_path,
        )
        expected_bytes = output.get("bytes")
        expected_digest = _fingerprint(
            output.get("sha256"),
            "selected output digest",
        )
        if (
            not isinstance(expected_bytes, int)
            or isinstance(expected_bytes, bool)
            or expected_bytes < 0
        ):
            raise ValueError("Canonical reportのselected output bytesが不正です")
        try:
            actual_bytes = artifact_path.stat().st_size
        except OSError:
            raise ValueError(
                "Completed runのselected output artifactを検証できません"
            ) from None
        if (
            actual_bytes != expected_bytes
            or _file_digest(artifact_path) != expected_digest
        ):
            raise ValueError("Completed runのselected output artifactが一致しません")


def _selected_output_relative_path(value: object) -> str:
    path = PurePosixPath(value) if isinstance(value, str) else None
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or path is None
        or path.is_absolute()
        or path.as_posix() != value
        or ".." in path.parts
    ):
        raise ValueError("Canonical reportのselected output pathが不正です")
    return value


def _selected_output_artifact_path(
    output_folder: Path,
    relative_path: str,
) -> Path:
    path = output_folder
    for part in PurePosixPath(relative_path).parts:
        path /= part
        if path.is_symlink():
            raise ValueError("Completed runのselected output artifactが不正です")
    if not path.is_file():
        raise ValueError("Completed runのselected output artifactがありません")
    return path


def _file_digest(path: Path) -> str:
    try:
        with path.open("rb") as file:
            return hashlib.file_digest(file, "sha256").hexdigest()
    except OSError:
        raise ValueError("Completed run artifactを読み込めません") from None


def normalized_result_digest(report: Mapping[str, object]) -> str:
    """cold/warm一致と永続証拠の再検証に使うdigestを返す。"""
    return canonical_report_semantic_digest(report)


def video_scan_parallelism_diagnostics(
    report: Mapping[str, object],
) -> dict[str, object]:
    """canonical reportからprivacy-safeなVideo Scan診断を返す。"""
    provenance = _mapping(
        report.get("provenance"),
        "canonical report provenance",
    )
    runtime = _mapping(
        provenance.get("runtime"),
        "canonical report provenance runtime",
    )
    return dict(
        _mapping(
            runtime.get("video_scan_parallelism"),
            "video_scan_parallelism",
        )
    )


def _mapping(value: object, location: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"Acceptance run {location}がobjectではありません")
    return cast(dict[str, object], value)


def _stage_fingerprint(value: object, location: str) -> StageFingerprint:
    return StageFingerprint(_fingerprint(value, location))


def _fingerprint(value: object, location: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"Acceptance run {location}が不正です")
    return value


def _video_set_record(report: Mapping[str, object]) -> dict[str, object]:
    video_set = report.get("video_set")
    if not isinstance(video_set, dict) or not isinstance(
        video_set.get("sources"),
        list,
    ):
        raise ValueError("Canonical reportのVideo Setが不正です")
    source_fingerprints: list[str] = []
    for source in video_set["sources"]:
        if not isinstance(source, dict) or not isinstance(
            source.get("fingerprint"),
            dict,
        ):
            raise ValueError("Canonical reportのVideo Sourceが不正です")
        fingerprint = source["fingerprint"].get("value")
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise ValueError("Canonical reportのVideo Fingerprintが不正です")
        source_fingerprints.append(fingerprint)
    fingerprint = hashlib.sha256()
    fingerprint.update(b"game-screen-pick/video-set-fingerprint@1\0")
    for source_fingerprint in source_fingerprints:
        fingerprint.update(bytes.fromhex(source_fingerprint))
    duration = video_set.get("duration")
    if not isinstance(duration, dict) or not isinstance(
        duration.get("exact_seconds"),
        str,
    ):
        raise ValueError("Canonical reportのVideo Set durationが不正です")
    return {
        "fingerprint": fingerprint.hexdigest(),
        "scenario_count": len(source_fingerprints),
        "total_duration_seconds": duration["exact_seconds"],
    }


def _speech_runtime_identity(report: Mapping[str, object]) -> str | None:
    provenance = report.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Canonical reportのprovenanceが不正です")
    runtime = provenance.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("Canonical reportのruntime provenanceが不正です")
    identity = runtime.get("speech_runtime_identity")
    if identity is None and "speech_runtime_identity" not in runtime:
        return None
    if not isinstance(identity, str) or not identity:
        raise ValueError("Canonical reportのSpeech Runtime Identityが不正です")
    return identity
