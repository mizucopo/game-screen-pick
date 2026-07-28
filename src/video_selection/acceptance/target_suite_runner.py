"""比較runとcold/warm phaseのresume、record、human gateを所有するrunner。"""

import hashlib
import json
import shutil
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import cast

from ..configuration.resolve_effective_configuration import (
    resolve_effective_configuration,
)
from ..model_runtime.model_lifecycle_runtime import ModelLifecycleRuntime
from ..models.effective_configuration import EffectiveConfiguration
from ..models.resolved_models import ResolvedModels
from .acceptance_execution_step import AcceptanceExecutionStep
from .acceptance_profile import AcceptanceProfile
from .acceptance_record import (
    build_acceptance_record,
    validate_acceptance_record_privacy,
    write_normalized_baseline,
)
from .acceptance_run import (
    AcceptanceRunAttemptExecutionResult,
    execute_acceptance_run_attempt,
    load_completed_run_evidence,
    load_completed_run_report,
    public_run_record,
)
from .acceptance_run_attempt_metrics import (
    aggregate_run_attempts,
    build_incomplete_interrupt_attempt,
    validate_run_measurements,
)
from .acceptance_storage_preflight import preflight_acceptance_storage
from .atomic_json import read_json_object, write_atomic_json
from .full_suite_materializer import FullSuiteMaterializer
from .human_review import (
    ensure_review_worksheet,
    evaluate_human_review,
    review_candidate_digest,
)
from .load_acceptance_profile import load_acceptance_profile
from .release_suite_materializer import ReleaseSuiteMaterializer
from .target_environment import (
    probe_source_revision,
    probe_target_environment,
    probe_windows_native_ollama,
)
from .video_scan_parallelism_comparison import (
    build_video_scan_parallelism_comparison,
)

EnvironmentProbe = Callable[[], dict[str, object]]
RevisionProbe = Callable[[Path], tuple[str, bool]]
OllamaDeploymentProbe = Callable[[str], dict[str, object]]
ModelResolver = Callable[[EffectiveConfiguration], ResolvedModels]
RunAttemptExecutor = Callable[
    [EffectiveConfiguration, ResolvedModels, Path],
    AcceptanceRunAttemptExecutionResult,
]
SuiteMaterializer = Callable[
    [AcceptanceProfile, Path],
    tuple[Path, dict[str, object]],
]
StoragePreflight = Callable[[AcceptanceProfile, Path], dict[str, object]]

_STATE_SCHEMA = "game-screen-pick/target-acceptance-state@1.2.0"
_VISIBLE_RAM_IDENTITY_TOLERANCE_BYTES = 1024**2


class TargetSuiteRunner:
    """比較run後にcold→exact warm→human gateをdurableに進める。"""

    def __init__(
        self,
        *,
        environment_probe: EnvironmentProbe = probe_target_environment,
        revision_probe: RevisionProbe = probe_source_revision,
        ollama_deployment_probe: OllamaDeploymentProbe = probe_windows_native_ollama,
        model_resolver: ModelResolver | None = None,
        run_attempt_executor: RunAttemptExecutor | None = None,
        release_materializer: SuiteMaterializer | None = None,
        full_materializer: SuiteMaterializer | None = None,
        storage_preflight: StoragePreflight = preflight_acceptance_storage,
    ) -> None:
        self._environment_probe = environment_probe
        self._revision_probe = revision_probe
        self._ollama_deployment_probe = ollama_deployment_probe
        self._model_resolver = model_resolver or ModelLifecycleRuntime().resolve_models
        self._run_attempt_executor = run_attempt_executor or _execute_run_attempt
        self._release_materializer = (
            release_materializer or ReleaseSuiteMaterializer().materialize
        )
        self._full_materializer = (
            full_materializer or FullSuiteMaterializer().materialize
        )
        self._storage_preflight = storage_preflight

    def run(
        self,
        *,
        profile_path: Path,
        suite: str,
        reset_suite: bool = False,
        human_review_path: Path | None = None,
    ) -> int:
        """suiteを未完了runから進めacceptance exit codeを返す。"""
        if suite not in {"release", "full"}:
            raise ValueError("--suiteにはreleaseまたはfullが必要です")
        profile = load_acceptance_profile(profile_path)
        _validate_profile_files(profile)
        suite_root = profile.artifact_root / "target-acceptance" / suite
        if reset_suite:
            _remove_directory_strict(suite_root, "Acceptance suite")
        state_path = suite_root / "acceptance-state.json"
        state = read_json_object(state_path)
        configuration_digest = _content_digest(profile.configuration_path)
        commit, dirty = self._revision_probe(Path.cwd())
        if dirty:
            raise ValueError("Target acceptanceはclean Git revisionで実行してください")
        if state is not None and (
            state.get("active_phase") is not None
            or state.get("active_comparison_run") is not None
        ):
            raise ValueError("前回runの計測が未確定です。--reset-suiteが必要です")

        target = self._environment_probe()
        input_folder, suite_descriptor = self._materialize(
            profile,
            suite,
            suite_root,
        )
        storage_preflight = (
            self._storage_preflight(profile, input_folder)
            if state is None
            else _mapping(state.get("storage_preflight"), "storage_preflight")
        )
        cold_configuration = _configuration(
            profile,
            input_folder,
            suite_root / "outputs" / "cold",
        )
        warm_configuration = _configuration(
            profile,
            input_folder,
            suite_root / "outputs" / "warm",
        )
        execution_steps = _execution_steps(
            suite,
            suite_root,
            cold_configuration,
            warm_configuration,
        )
        if state is not None and state.get(
            "ollama_endpoint_identity"
        ) != _ollama_endpoint_identity(cold_configuration.ollama_host):
            raise ValueError("Acceptance stateが現在のsuite identityと一致しません")
        ollama_deployment = self._ollama_deployment_probe(
            cold_configuration.ollama_host
        )
        target = {**target, "ollama": ollama_deployment}
        try:
            resolved_models = self._model_resolver(cold_configuration)
        except KeyboardInterrupt:
            return 130
        except Exception:
            return 1
        if (
            self._ollama_deployment_probe(cold_configuration.ollama_host)
            != ollama_deployment
        ):
            raise ValueError("Windows Ollama bindingがmodel解決中に変更されました")
        identity = _identity(
            suite,
            profile,
            configuration_digest,
            cold_configuration,
            suite_descriptor,
            resolved_models,
            commit,
        )
        if state is None:
            state = {
                "schema": _STATE_SCHEMA,
                **identity,
                "source_revision": {"commit": commit, "dirty": False},
                "target": target,
                "configuration": _configuration_summary(
                    cold_configuration,
                    configuration_digest=configuration_digest,
                ),
                "models": resolved_models.provenance(),
                "storage_preflight": storage_preflight,
                "phases": {},
            }
            if suite == "full":
                state["comparison_runs"] = {}
            write_atomic_json(state_path, state)
        else:
            _validate_state_identity(state, identity)
            if not _target_identity_matches(state.get("target"), target):
                raise ValueError(
                    "Acceptance stateが現在のtarget identityと一致しません"
                )

        if _runs_completed(state) and state.get("worksheet_ready") is True:
            for step in execution_steps:
                completed_runs = _mapping(
                    state.get(step.records_state_key),
                    step.records_state_key,
                )
                load_completed_run_report(
                    configuration=step.configuration,
                    run_record=_mapping(
                        completed_runs.get(step.name),
                        f"{step.name} run",
                    ),
                )
            return self._finalize(
                profile,
                suite_root,
                state,
                human_review_path=human_review_path,
            )

        cold_report: dict[str, object] | None = None
        cold_selection: dict[str, object] | None = None
        for step in execution_steps:
            runs = _mapping(
                state.get(step.records_state_key),
                step.records_state_key,
            )
            if step.is_cold_phase and suite == "full":
                _release_fixed_three_cache(
                    state,
                    cold_configuration.processing_cache_folder,
                    state_path,
                )
            existing = runs.get(step.name)
            if (
                isinstance(existing, dict)
                and existing.get("operation_status") == "completed"
            ):
                continue
            shutil.rmtree(step.configuration.output_folder, ignore_errors=True)
            state[step.active_state_key] = step.name
            write_atomic_json(state_path, state)
            attempt_started_at = time.monotonic()
            try:
                (
                    exit_code,
                    attempt_record,
                    report,
                    selection,
                ) = self._run_attempt_executor(
                    step.configuration,
                    resolved_models,
                    suite_root,
                )
            except KeyboardInterrupt:
                exit_code = 130
                attempt_record = build_incomplete_interrupt_attempt(
                    time.monotonic() - attempt_started_at
                )
                report = None
                selection = None
            except Exception:
                state["last_failure"] = {
                    **step.failure_context,
                    "exit_code": 1,
                    "reason": "run_measurement_incomplete",
                }
                write_atomic_json(state_path, state)
                return 1
            validate_run_measurements(attempt_record)
            prior_attempts = _run_attempts(state, step)
            state.pop(step.active_state_key, None)
            if exit_code != 0:
                _store_run_attempts(
                    state,
                    step,
                    (*prior_attempts, attempt_record),
                )
                state["last_failure"] = {
                    **step.failure_context,
                    "exit_code": exit_code,
                    "reason": attempt_record.get(
                        "failure_reason",
                        "operation_failed",
                    ),
                }
                write_atomic_json(state_path, state)
                return exit_code
            if attempt_record.get("operation_status") != "completed":
                raise ValueError("成功runの計測記録がcompletedではありません")
            runs[step.name] = aggregate_run_attempts((*prior_attempts, attempt_record))
            state[step.records_state_key] = runs
            _clear_run_attempts(state, step)
            state.pop("last_failure", None)
            write_atomic_json(state_path, state)
            if step.is_cold_phase:
                cold_report = report
                cold_selection = selection

        phases = _mapping(state.get("phases"), "phases")
        if cold_report is None or cold_selection is None:
            cold_phase = _mapping(phases.get("cold"), "cold phase")
            cold_report, cold_selection = load_completed_run_evidence(
                configuration=cold_configuration,
                run_record=cold_phase,
            )
        cold_video_set = _mapping(
            _mapping(phases.get("cold"), "cold phase").get("video_set"),
            "video_set",
        )
        state["video_set"] = cold_video_set
        worksheet = ensure_review_worksheet(
            suite_root / "review-worksheet.json",
            suite=suite,
            suite_fingerprint=_string(state.get("suite_fingerprint")),
            canonical_report=cold_report,
            selection_artifact=cold_selection,
        )
        state["review_candidate_digest"] = review_candidate_digest(worksheet)
        state["worksheet_ready"] = True
        write_atomic_json(state_path, state)
        return self._finalize(
            profile,
            suite_root,
            state,
            human_review_path=human_review_path,
        )

    def _materialize(
        self,
        profile: AcceptanceProfile,
        suite: str,
        suite_root: Path,
    ) -> tuple[Path, dict[str, object]]:
        if suite == "release":
            return self._release_materializer(profile, suite_root)
        return self._full_materializer(profile, suite_root)

    def _finalize(
        self,
        profile: AcceptanceProfile,
        suite_root: Path,
        state: dict[str, object],
        *,
        human_review_path: Path | None,
    ) -> int:
        """record、baseline、release cleanupをphase再実行なしで確定する。"""
        suite = _string(state.get("suite"))
        worksheet_path = human_review_path or suite_root / "review-worksheet.json"
        worksheet = read_json_object(worksheet_path)
        if worksheet is None:
            raise ValueError("Human review worksheetが見つかりません")
        human_quality = evaluate_human_review(
            worksheet,
            suite=suite,
            suite_fingerprint=_string(state.get("suite_fingerprint")),
            expected_candidate_digest=_string(state.get("review_candidate_digest")),
        )
        phases = _mapping(state.get("phases"), "phases")
        cold = _mapping(phases.get("cold"), "cold phase")
        warm = _mapping(phases.get("warm"), "warm phase")
        parallelism_comparison = (
            None
            if suite != "full"
            else build_video_scan_parallelism_comparison(
                _mapping(
                    _mapping(
                        state.get("comparison_runs"),
                        "comparison_runs",
                    ).get("fixed3"),
                    "fixed3 comparison run",
                ),
                cold,
            )
        )
        revision = _mapping(state.get("source_revision"), "source_revision")
        record = build_acceptance_record(
            suite=suite,
            commit=_string(revision.get("commit")),
            dirty=_boolean(revision.get("dirty")),
            target=_mapping(state.get("target"), "target"),
            configuration=_mapping(state.get("configuration"), "configuration"),
            models=_mapping(state.get("models"), "models"),
            storage_preflight=_mapping(
                state.get("storage_preflight"),
                "storage_preflight",
            ),
            video_set=_mapping(state.get("video_set"), "video_set"),
            cold=public_run_record(cold),
            warm=public_run_record(warm),
            human_quality=human_quality,
            video_scan_parallelism_comparison=parallelism_comparison,
        )
        try:
            validate_acceptance_record_privacy(
                record,
                forbidden_values=_forbidden_values(profile),
            )
        except ValueError:
            state["acceptance_status"] = "failed"
            state["last_failure"] = {
                "phase": "acceptance_record",
                "exit_code": 1,
                "reason": "privacy_gate_failed",
            }
            if not _cleanup_release_work(
                suite,
                suite_root,
                state,
                prior_failure_reason="privacy_gate_failed",
            ):
                return 1
            write_atomic_json(suite_root / "acceptance-state.json", state)
            return 1
        if not _cleanup_release_work(suite, suite_root, state):
            return 1
        _remove_directory_strict(suite_root / "baseline", "Acceptance baseline")
        write_atomic_json(suite_root / "acceptance.json", record)
        state["acceptance_status"] = record["status"]
        state.pop("last_failure", None)
        write_atomic_json(suite_root / "acceptance-state.json", state)
        if record["status"] == "passed":
            write_normalized_baseline(record, suite_root / "baseline")
        return (
            0
            if record["status"] == "passed"
            else 3
            if record["status"] == "pending_human_review"
            else 1
        )


def _cleanup_release_work(
    suite: str,
    suite_root: Path,
    state: dict[str, object],
    *,
    prior_failure_reason: str | None = None,
) -> bool:
    """release private workを完全に削除できた場合だけfinalizationを許可する。"""
    if suite != "release":
        return True
    try:
        _remove_directory_strict(suite_root / "work", "Release acceptance work")
    except ValueError:
        failure: dict[str, object] = {
            "phase": "acceptance_cleanup",
            "exit_code": 1,
            "reason": "release_cleanup_failed",
        }
        if prior_failure_reason is not None:
            failure["prior_reason"] = prior_failure_reason
        state["acceptance_status"] = "failed"
        state["last_failure"] = failure
        write_atomic_json(suite_root / "acceptance-state.json", state)
        return False
    return True


def _remove_directory_strict(path: Path, label: str) -> None:
    """reset対象directoryが完全に削除された場合だけ後続処理を許可する。"""
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink():
        raise ValueError(f"{label}はsymbolic linkのため削除できません")
    try:
        shutil.rmtree(path)
    except OSError:
        raise ValueError(f"{label}を完全に削除できません") from None
    if path.exists() or path.is_symlink():
        raise ValueError(f"{label}を完全に削除できません")


def _execute_run_attempt(
    configuration: EffectiveConfiguration,
    resolved_models: ResolvedModels,
    suite_root: Path,
) -> AcceptanceRunAttemptExecutionResult:
    return execute_acceptance_run_attempt(
        configuration=configuration,
        resolved_models=resolved_models,
        suite_root=suite_root,
    )


def _configuration(
    profile: AcceptanceProfile,
    input_folder: Path,
    output_folder: Path,
) -> EffectiveConfiguration:
    return resolve_effective_configuration(
        video_input_folder=input_folder,
        output_folder=output_folder,
        config_path=profile.configuration_path,
    )


def _execution_steps(
    suite: str,
    suite_root: Path,
    cold: EffectiveConfiguration,
    warm: EffectiveConfiguration,
) -> tuple[AcceptanceExecutionStep, ...]:
    if suite != "full":
        return (
            AcceptanceExecutionStep("phase", "cold", cold),
            AcceptanceExecutionStep("phase", "warm", warm),
        )
    if cold.video_scan_workers != "auto":
        raise ValueError("Full acceptanceのVideo Scan workersにはautoが必要です")
    fixed_three = replace(
        cold,
        output_folder=suite_root / "outputs" / "fixed3",
        video_scan_workers=3,
    )
    return (
        AcceptanceExecutionStep("comparison", "fixed3", fixed_three),
        AcceptanceExecutionStep("phase", "cold", cold),
        AcceptanceExecutionStep("phase", "warm", warm),
    )


def _release_fixed_three_cache(
    state: dict[str, object],
    cache_folder: Path,
    state_path: Path,
) -> None:
    if state.get("fixed3_cache_released") is True:
        return
    comparison_runs = _mapping(state.get("comparison_runs"), "comparison_runs")
    fixed_three = _mapping(
        comparison_runs.get("fixed3"),
        "fixed3 comparison run",
    )
    if fixed_three.get("operation_status") != "completed":
        raise ValueError("Fixed3 comparison完了前にauto cacheを開始できません")
    _remove_directory_strict(cache_folder, "Fixed3 comparison cache")
    state["fixed3_cache_released"] = True
    write_atomic_json(state_path, state)


def _configuration_summary(
    configuration: EffectiveConfiguration,
    *,
    configuration_digest: str,
) -> dict[str, object]:
    return {
        "configuration_digest": configuration_digest,
        "config_version": configuration.config_version,
        "recursive": configuration.recursive,
        "image_count": configuration.image_count,
        "scene_hint_identity": _optional_text_identity(
            "scene-hint",
            configuration.scene_hint,
        ),
        "spoiler_sensitivity": configuration.spoiler_sensitivity,
        "similarity_threshold": configuration.similarity_threshold,
        "heartbeat_interval_seconds": configuration.heartbeat_interval_seconds,
        "scene_change_threshold": configuration.scene_change_threshold,
        "scene_min_interval_seconds": configuration.scene_min_interval_seconds,
        "decode_backend": configuration.decode_backend,
        "video_scan_workers": configuration.video_scan_workers,
        "video_scan_auto_max_workers": (configuration.video_scan_auto_max_workers),
        "refinement_radius_seconds": configuration.refinement_radius_seconds,
        "max_frame_candidates": configuration.max_frame_candidates,
        "candidate_density_per_minute": configuration.candidate_density_per_minute,
        "language": configuration.language,
        "subtitle_stream_index": configuration.subtitle_stream_index,
        "audio_stream_index": configuration.audio_stream_index,
        "ollama_timeout_seconds": configuration.ollama_timeout_seconds,
        "ollama_max_parallel_requests": configuration.ollama_max_parallel_requests,
        "models_auto_upgrade": configuration.models_auto_upgrade,
        "scene_catalog_model": configuration.scene_catalog_model,
        "scene_catalog_num_ctx": configuration.scene_catalog_num_ctx,
        "candidate_annotation_model": configuration.candidate_annotation_model,
        "candidate_annotation_num_ctx": configuration.candidate_annotation_num_ctx,
        "speech_to_text_model": configuration.speech_to_text_model,
        "speech_to_text_device": configuration.speech_to_text_device,
        "speech_to_text_compute_type": configuration.speech_to_text_compute_type,
        "speech_to_text_beam_size": configuration.speech_to_text_beam_size,
        "speech_vad_filter": configuration.speech_vad_filter,
        "speech_chunk_seconds": configuration.speech_chunk_seconds,
        "speech_overlap_seconds": configuration.speech_overlap_seconds,
        "reset_cache": configuration.reset_cache,
        "debug": configuration.debug,
        "ollama_endpoint_identity": _ollama_endpoint_identity(
            configuration.ollama_host
        ),
    }


def _identity(
    suite: str,
    profile: AcceptanceProfile,
    configuration_digest: str,
    configuration: EffectiveConfiguration,
    descriptor: Mapping[str, object],
    models: ResolvedModels,
    commit: str,
) -> dict[str, object]:
    descriptor_fingerprint = descriptor.get(
        "suite_fingerprint",
        descriptor.get("source_snapshot_fingerprint"),
    )
    if not isinstance(descriptor_fingerprint, str):
        raise ValueError("Suite materialization fingerprintがありません")
    models_json = json.dumps(
        models.semantic_input(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {
        "suite": suite,
        "profile_digest": profile.profile_digest,
        "configuration_digest": configuration_digest,
        "ollama_endpoint_identity": _ollama_endpoint_identity(
            configuration.ollama_host
        ),
        "suite_fingerprint": descriptor_fingerprint,
        "model_identity_digest": hashlib.sha256(models_json).hexdigest(),
        "commit": commit,
    }


def _validate_profile_files(profile: AcceptanceProfile) -> None:
    if not profile.input_root.is_dir():
        raise ValueError("Acceptance input rootが存在しません")
    if not profile.configuration_path.is_file():
        raise ValueError("Acceptance configurationが存在しません")
    profile.artifact_root.mkdir(parents=True, exist_ok=True)


def _validate_state_identity(
    state: Mapping[str, object],
    identity: Mapping[str, object],
) -> None:
    if state.get("schema") != _STATE_SCHEMA or any(
        state.get(key) != value for key, value in identity.items()
    ):
        raise ValueError("Acceptance stateが現在のsuite identityと一致しません")


def _target_identity_matches(
    stored: object,
    current: Mapping[str, object],
) -> bool:
    """起動時の微小なvisible RAM差だけを許容してtargetを比較する。"""
    if not isinstance(stored, Mapping) or set(stored) != set(current):
        return False
    stored_ram = stored.get("visible_ram_bytes")
    current_ram = current.get("visible_ram_bytes")
    if (
        type(stored_ram) is not int
        or type(current_ram) is not int
        or stored_ram <= 0
        or current_ram <= 0
        or abs(stored_ram - current_ram) > _VISIBLE_RAM_IDENTITY_TOLERANCE_BYTES
    ):
        return False
    return all(
        stored.get(key) == value
        for key, value in current.items()
        if key != "visible_ram_bytes"
    )


def _runs_completed(state: Mapping[str, object]) -> bool:
    phases = state.get("phases")
    if not isinstance(phases, dict):
        return False
    phases_completed = all(
        isinstance(phases.get(phase), dict)
        and phases[phase].get("operation_status") == "completed"
        for phase in ("cold", "warm")
    )
    if not phases_completed or state.get("suite") != "full":
        return phases_completed
    comparison_runs = state.get("comparison_runs")
    return (
        isinstance(comparison_runs, dict)
        and isinstance(comparison_runs.get("fixed3"), dict)
        and comparison_runs["fixed3"].get("operation_status") == "completed"
    )


def _run_attempts(
    state: Mapping[str, object],
    step: AcceptanceExecutionStep,
) -> tuple[dict[str, object], ...]:
    attempts = state.get(step.attempts_state_key)
    if attempts is None:
        return ()
    attempts_by_run = _mapping(attempts, step.attempts_label)
    values = attempts_by_run.get(step.name, [])
    if not isinstance(values, list) or any(
        not isinstance(value, dict) for value in values
    ):
        raise ValueError(f"Acceptance state {step.attempts_label}が不正です")
    return tuple(cast(dict[str, object], value) for value in values)


def _store_run_attempts(
    state: dict[str, object],
    step: AcceptanceExecutionStep,
    attempts: tuple[Mapping[str, object], ...],
) -> None:
    existing = state.get(step.attempts_state_key)
    attempts_by_run = (
        {} if existing is None else _mapping(existing, step.attempts_label)
    )
    attempts_by_run[step.name] = [dict(attempt) for attempt in attempts]
    state[step.attempts_state_key] = attempts_by_run


def _clear_run_attempts(
    state: dict[str, object],
    step: AcceptanceExecutionStep,
) -> None:
    existing = state.get(step.attempts_state_key)
    if existing is None:
        return
    attempts_by_run = _mapping(existing, step.attempts_label)
    attempts_by_run.pop(step.name, None)
    if attempts_by_run:
        state[step.attempts_state_key] = attempts_by_run
    else:
        state.pop(step.attempts_state_key, None)


def _forbidden_values(profile: AcceptanceProfile) -> tuple[str, ...]:
    return (
        str(profile.input_root),
        str(profile.configuration_path),
        str(profile.artifact_root),
        *(item.relative_video_path for item in profile.release_intervals),
        *(Path(item.relative_video_path).name for item in profile.release_intervals),
    )


def _content_digest(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _ollama_endpoint_identity(host: str) -> str:
    normalized = host.rstrip("/")
    payload = f"game-screen-pick/ollama-endpoint@1\0{normalized}".encode()
    return hashlib.sha256(payload).hexdigest()


def _optional_text_identity(label: str, value: str | None) -> str | None:
    if value is None:
        return None
    payload = f"game-screen-pick/{label}@1\0{value}".encode()
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: object, location: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"Acceptance state {location}がobjectではありません")
    return cast(dict[str, object], value)


def _string(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("Acceptance state valueがstringではありません")
    return value


def _boolean(value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError("Acceptance state valueがbooleanではありません")
    return value
