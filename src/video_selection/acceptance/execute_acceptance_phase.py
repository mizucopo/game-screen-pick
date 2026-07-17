"""一つのcold/warm acceptance phaseを実pipelineで実行・計測する。"""

import hashlib
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from ..application.internal_run_controller import InternalRunController
from ..models.effective_configuration import EffectiveConfiguration
from ..models.processing_stage import ProcessingStage
from ..models.resolved_models import ResolvedModels
from ..models.run_failure import RunFailure
from ..models.run_outcome import RunOutcome
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
        disk_metrics = disk_monitor.stop()
        gpu_metrics = gpu_monitor.stop()
    duration_seconds = time.monotonic() - started_at
    phase_record: dict[str, object] = {
        "duration_seconds": duration_seconds,
        **observer.phase_metrics(),
        **disk_metrics,
        **gpu_metrics,
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
    report = _read_json_object(result.output_folder / "report.json")
    selection_stage = next(
        stage
        for stage in reversed(result.completed_stages)
        if stage.stage is ProcessingStage.SELECT_IMAGES
    )
    selection_artifact = _selection_artifact(
        configuration.processing_cache_folder,
        selection_stage.fingerprint.value,
    )
    phase_record.update(
        {
            "operation_status": "completed",
            "selected_count": result.selected_count,
            "requested_count": result.requested_count,
            "normalized_result_digest": _normalized_result_digest(report),
            "selection_stage_fingerprint": selection_stage.fingerprint.value,
            "video_set": _video_set_record(report),
        }
    )
    return 0, phase_record, report, selection_artifact


def load_completed_phase_evidence(
    *,
    configuration: EffectiveConfiguration,
    phase_record: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """durable output/cacheからcompleted phaseのworksheet sourceを復元する。"""
    report = _read_json_object(configuration.output_folder / "report.json")
    fingerprint = phase_record.get("selection_stage_fingerprint")
    if not isinstance(fingerprint, str):
        raise ValueError("Completed phaseにselection fingerprintがありません")
    return report, _selection_artifact(
        configuration.processing_cache_folder,
        fingerprint,
    )


def public_phase_record(value: Mapping[str, object]) -> dict[str, object]:
    """suite stateのphaseからrun-level Video Set重複値を除いて返す。"""
    return {key: item for key, item in value.items() if key != "video_set"}


def _selection_artifact(cache_folder: Path, fingerprint: str) -> dict[str, object]:
    matches = tuple(
        cache_folder.glob(f"video-sets/*/select-images/{fingerprint}/artifact.json")
    )
    if len(matches) != 1:
        raise ValueError("Selection Stage artifactを一意に復元できません")
    return _read_json_object(matches[0])


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        raise ValueError("Acceptance phase artifactを読み込めません") from None
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError("Acceptance phase artifactがobjectではありません")
    return cast(dict[str, object], value)


def _normalized_result_digest(report: Mapping[str, object]) -> str:
    selected = report.get("selected")
    provenance = report.get("provenance")
    if not isinstance(selected, list) or not isinstance(provenance, dict):
        raise ValueError("Canonical reportのselected/provenanceが不正です")
    normalized_selected: list[dict[str, object]] = []
    for value in selected:
        if not isinstance(value, dict):
            raise ValueError("Canonical reportのselected recordが不正です")
        normalized_selected.append(
            {
                key: value.get(key)
                for key in (
                    "image_id",
                    "selection_index",
                    "classification",
                    "annotation",
                    "selection",
                )
            }
        )
    normalized = {
        "selected": normalized_selected,
        "models": provenance.get("models"),
    }
    canonical = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


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
