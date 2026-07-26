"""一つのcold/warm acceptance phaseを実pipelineで実行・計測する。"""

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
from ..services.completed_stage_writer import CompletedStageWriter
from ..services.progress_stream_observer import ProgressStreamObserver
from ..services.run_progress_tracker import RunProgressTracker
from .acceptance_run_observer import AcceptanceRunObserver
from .build_real_application import build_real_application
from .disk_usage_monitor import DiskUsageMonitor
from .gpu_resource_monitor import GpuResourceMonitor

PhaseExecutionResult = tuple[
    int,
    dict[str, object],
    dict[str, object] | None,
    dict[str, object] | None,
]
_VOLATILE_RUN_KEYS = frozenset({"id", "started_at", "completed_at"})
_VOLATILE_STAGE_DIAGNOSTIC_KEYS = frozenset(
    {
        "attempt_count",
        "cache_hits",
        "cache_misses",
        "duration_ms",
        "eval_tokens",
        "prompt_eval_tokens",
        "recomputed_items",
        "validation_failures",
    }
)
_VOLATILE_MODEL_LIFECYCLE_KEYS = frozenset(
    {
        "local_identity_before_update",
        "update_status",
    }
)


def execute_acceptance_phase(
    *,
    phase: str,
    configuration: EffectiveConfiguration,
    resolved_models: ResolvedModels,
    suite_root: Path,
) -> PhaseExecutionResult:
    """model freeze後からatomic publicationまでを測りsafe evidenceを返す。"""
    if phase not in {"cold", "warm"}:
        raise ValueError("Acceptance phaseが不正です")
    observer = AcceptanceRunObserver(ProgressStreamObserver())
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
        phase_completed_at = time.monotonic()
        disk_metrics = disk_monitor.stop()
        gpu_metrics = gpu_monitor.stop()
    duration_seconds = phase_completed_at - started_at
    phase_record: dict[str, object] = {
        "duration_seconds": duration_seconds,
        **observer.phase_metrics(),
        **disk_metrics,
        **gpu_metrics,
        "resource_sampling_complete": (
            disk_metrics.get("disk_sampling_complete") is True
            and gpu_metrics.get("resource_sampling_complete") is True
        ),
    }
    if exit_code != 0:
        if not isinstance(result, RunFailure):
            raise AssertionError
        phase_record.update(
            {
                "operation_status": "failed",
                "failure_reason": result.reason_code,
                "failure_exit_code": result.exit_code,
            }
        )
        return int(exit_code), phase_record, None, None
    if not isinstance(result, RunOutcome):
        raise AssertionError
    report_path = result.output_folder / "report.json"
    report = _read_json_object(report_path)
    selection_stage = next(
        stage
        for stage in reversed(result.completed_stages)
        if stage.stage is ProcessingStage.SELECT_IMAGES
    )
    phase_record.update(
        {
            "operation_status": "completed",
            "selected_count": result.selected_count,
            "requested_count": result.requested_count,
            "canonical_report_sha256": _file_digest(report_path),
            "normalized_result_digest": normalized_result_digest(report),
            "selection_stage_fingerprint": selection_stage.fingerprint.value,
            "video_set": _video_set_record(report),
            "speech_runtime_identity": _speech_runtime_identity(report),
        }
    )
    selection_artifact = _selection_artifact(
        configuration.processing_cache_folder,
        phase_record,
    )
    return 0, phase_record, report, selection_artifact


def load_completed_phase_evidence(
    *,
    configuration: EffectiveConfiguration,
    phase_record: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """durable output/cacheからcompleted phaseのworksheet sourceを復元する。"""
    report = load_completed_phase_report(
        configuration=configuration,
        phase_record=phase_record,
    )
    return report, _selection_artifact(
        configuration.processing_cache_folder,
        phase_record,
    )


def load_completed_phase_report(
    *,
    configuration: EffectiveConfiguration,
    phase_record: Mapping[str, object],
) -> dict[str, object]:
    """durable canonical reportと公開画像がphase確定時のままか再検証する。"""
    output_folder = configuration.output_folder
    report_path = output_folder / "report.json"
    if (
        output_folder.is_symlink()
        or not output_folder.is_dir()
        or report_path.is_symlink()
        or not report_path.is_file()
    ):
        raise ValueError("Completed phaseのcanonical report artifactがありません")
    expected_report_digest = _fingerprint(
        phase_record.get("canonical_report_sha256"),
        "canonical report digest",
    )
    if _file_digest(report_path) != expected_report_digest:
        raise ValueError("Completed phaseのcanonical report artifactが一致しません")
    report = _read_json_object(report_path)
    expected_digest = phase_record.get("normalized_result_digest")
    if (
        not isinstance(expected_digest, str)
        or normalized_result_digest(report) != expected_digest
    ):
        raise ValueError("Completed phaseのcanonical report digestが一致しません")
    _validate_selected_output_artifacts(report, output_folder)
    return report


def public_phase_record(value: Mapping[str, object]) -> dict[str, object]:
    """suite stateのphaseからrun-level Video Set重複値を除いて返す。"""
    private_state_keys = {"canonical_report_sha256", "video_set"}
    return {key: item for key, item in value.items() if key not in private_state_keys}


def _selection_artifact(
    cache_folder: Path,
    phase_record: Mapping[str, object],
) -> dict[str, object]:
    fingerprint = _stage_fingerprint(
        phase_record.get("selection_stage_fingerprint"),
        "selection fingerprint",
    )
    video_set = _mapping(phase_record.get("video_set"), "Video Set")
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
        raise ValueError("Acceptance phase artifactを読み込めません") from None
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError("Acceptance phase artifactがobjectではありません")
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
                "Completed phaseのselected output artifactを検証できません"
            ) from None
        if (
            actual_bytes != expected_bytes
            or _file_digest(artifact_path) != expected_digest
        ):
            raise ValueError("Completed phaseのselected output artifactが一致しません")


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
            raise ValueError("Completed phaseのselected output artifactが不正です")
    if not path.is_file():
        raise ValueError("Completed phaseのselected output artifactがありません")
    return path


def _file_digest(path: Path) -> str:
    try:
        with path.open("rb") as file:
            return hashlib.file_digest(file, "sha256").hexdigest()
    except OSError:
        raise ValueError("Completed phase artifactを読み込めません") from None


def normalized_result_digest(report: Mapping[str, object]) -> str:
    """cold/warm一致と永続証拠の再検証に使うdigestを返す。"""

    normalized = _normalized_semantic_report(report)
    canonical = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _normalized_semantic_report(
    report: Mapping[str, object],
) -> dict[str, object]:
    """run固有診断だけを除いたcanonical report全体を返す。"""
    run = _mapping(report.get("run"), "canonical report run")
    provenance = report.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Canonical reportのprovenanceが不正です")
    stages = provenance.get("stages")
    if not isinstance(stages, list):
        raise ValueError("Canonical reportのprovenance stagesが不正です")
    models = _mapping(
        provenance.get("models"),
        "canonical report provenance models",
    )
    normalized = dict(report)
    normalized["run"] = {
        key: value for key, value in run.items() if key not in _VOLATILE_RUN_KEYS
    }
    normalized["provenance"] = {
        **provenance,
        "models": {
            role: {
                key: item
                for key, item in _mapping(
                    model,
                    "canonical report provenance model",
                ).items()
                if key not in _VOLATILE_MODEL_LIFECYCLE_KEYS
            }
            for role, model in models.items()
        },
        "stages": [
            {
                key: item
                for key, item in _mapping(
                    value,
                    "canonical report provenance stage",
                ).items()
                if key not in _VOLATILE_STAGE_DIAGNOSTIC_KEYS
            }
            for value in stages
        ],
    }
    return normalized


def _mapping(value: object, location: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"Acceptance phase {location}がobjectではありません")
    return cast(dict[str, object], value)


def _stage_fingerprint(value: object, location: str) -> StageFingerprint:
    return StageFingerprint(_fingerprint(value, location))


def _fingerprint(value: object, location: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"Acceptance phase {location}が不正です")
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


def _speech_runtime_identity(report: Mapping[str, object]) -> str:
    provenance = report.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Canonical reportのprovenanceが不正です")
    runtime = provenance.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("Canonical reportのruntime provenanceが不正です")
    identity = runtime.get("speech_runtime_identity")
    if not isinstance(identity, str) or not identity:
        raise ValueError("Canonical reportにSpeech Runtime Identityがありません")
    return identity
