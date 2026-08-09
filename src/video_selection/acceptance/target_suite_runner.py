"""比較runとcold/warm phaseのresume、record、human gateを所有するrunner。"""

import hashlib
import json
import shutil
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from typing import cast
from uuid import uuid4

from ..configuration.resolve_effective_configuration import (
    resolve_effective_configuration,
)
from ..model_runtime.model_lifecycle_runtime import ModelLifecycleRuntime
from ..models.effective_configuration import EffectiveConfiguration
from ..models.resolved_models import ResolvedModels
from ..services.adaptive_video_scan_controller import AdaptiveVideoScanController
from ..services.validate_canonical_selection_report import (
    load_validated_canonical_selection_report,
)
from .acceptance_attempt_journal import AcceptanceAttemptJournal
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
from .acceptance_run_reset import ACCEPTANCE_RUN_RESETS, AcceptanceRunReset
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
from .source_snapshot_fingerprint import acceptance_source_snapshot_fingerprint
from .target_environment import (
    probe_source_revision,
    probe_target_environment,
    probe_windows_native_ollama,
)
from .video_scan_parallelism_comparison import (
    acceptance_run_matches_evidence_context,
    build_video_scan_parallelism_comparison,
    video_scan_run_matches_comparison_context,
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

_STATE_SCHEMA = "game-screen-pick/target-acceptance-state@1.3.0"
_ACTIVE_RUN_STATE_KEYS = ("active_phase", "active_comparison_run")


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
        reset_run: AcceptanceRunReset | None = None,
        human_review_path: Path | None = None,
    ) -> int:
        """suiteを未完了runから進めacceptance exit codeを返す。"""
        if suite not in {"release", "full"}:
            raise ValueError("--suiteにはreleaseまたはfullが必要です")
        if reset_run is not None and reset_run not in ACCEPTANCE_RUN_RESETS:
            raise ValueError("--reset-runの対象が不正です")
        if reset_suite and reset_run is not None:
            raise ValueError("--reset-suiteと--reset-runは同時指定できません")
        if reset_run == "parallelism-baseline" and suite != "full":
            raise ValueError("parallelism-baseline resetはfull suiteだけで利用できます")
        profile = load_acceptance_profile(profile_path)
        _validate_profile_files(profile)
        suite_root = profile.artifact_root / "target-acceptance" / suite
        _validate_suite_source_paths(profile_path, profile, suite_root)
        if reset_suite:
            _remove_directory_strict(suite_root, "Acceptance suite")
        state_path = suite_root / "acceptance-state.json"
        attempt_journal = AcceptanceAttemptJournal(
            suite_root / "work" / "active-attempt.json"
        )
        state = read_json_object(state_path)
        if state is None:
            attempt_journal.clear()
        if reset_run == "cache-reuse":
            _validate_cache_reuse_reset_prerequisites(
                state,
                suite_root / "work" / "input" / ".game-screen-pick" / "cache",
            )
        configuration_digest = _content_digest(profile.configuration_path)
        identity_cache_folder = suite_root.parent / "video-identities"
        if (
            state is not None
            and reset_run is None
            and _runs_completed(state)
            and state.get("worksheet_ready") is True
            and _is_sha256(state.get("materialization_source_snapshot_fingerprint"))
        ):
            _validate_completed_state_identity(
                state,
                suite,
                profile,
                acceptance_source_snapshot_fingerprint(profile, suite),
            )
            input_folder = suite_root / "work" / "input"
            cold_configuration = _configuration(
                profile,
                input_folder,
                suite_root / "outputs" / "cold",
                identity_cache_folder,
            )
            warm_configuration = _configuration(
                profile,
                input_folder,
                suite_root / "outputs" / "warm",
                identity_cache_folder,
            )
            execution_steps = _execution_steps(
                suite,
                suite_root,
                cold_configuration,
                warm_configuration,
            )
            if any(
                state.get(step.active_state_key) is not None for step in execution_steps
            ):
                raise ValueError("完了済みAcceptance stateにactive runがあります")
            attempt_journal.clear()
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
        input_folder, suite_descriptor = self._materialize(
            profile,
            suite,
            suite_root,
        )
        cold_configuration = _configuration(
            profile,
            input_folder,
            suite_root / "outputs" / "cold",
            identity_cache_folder,
        )
        warm_configuration = _configuration(
            profile,
            input_folder,
            suite_root / "outputs" / "warm",
            identity_cache_folder,
        )
        configuration_summary = _configuration_summary(
            cold_configuration,
            configuration_digest=configuration_digest,
        )
        effective_configuration_digest = _effective_configuration_digest(
            configuration_summary
        )
        execution_steps = _execution_steps(
            suite,
            suite_root,
            cold_configuration,
            warm_configuration,
        )
        descriptor_fingerprint = suite_descriptor.get(
            "suite_fingerprint",
            suite_descriptor.get("source_snapshot_fingerprint"),
        )
        materialization_source_snapshot = suite_descriptor.get(
            "source_snapshot_fingerprint"
        )
        if not isinstance(descriptor_fingerprint, str):
            raise ValueError("Suite materialization fingerprintがありません")
        if not _is_sha256(materialization_source_snapshot):
            raise ValueError("Suite source snapshot fingerprintがありません")
        if state is not None:
            if "materialization_source_snapshot_fingerprint" not in state:
                state["materialization_source_snapshot_fingerprint"] = (
                    materialization_source_snapshot
                )
                write_atomic_json(state_path, state)
            _validate_state_identity(
                state,
                {
                    "suite": suite,
                    "profile_digest": profile.profile_digest,
                    "suite_fingerprint": descriptor_fingerprint,
                    "materialization_source_snapshot_fingerprint": (
                        materialization_source_snapshot
                    ),
                },
            )
            _recover_abandoned_attempt(
                state,
                execution_steps,
                state_path,
                attempt_journal,
            )
            attempt_journal.clear()
            if reset_run is not None:
                _reset_acceptance_run_suffix(
                    reset_run=reset_run,
                    state=state,
                    execution_steps=execution_steps,
                    suite_root=suite_root,
                    processing_cache_folder=(
                        cold_configuration.processing_cache_folder
                    ),
                    state_path=state_path,
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

        commit, dirty = self._revision_probe(Path.cwd())
        if dirty:
            raise ValueError("Target acceptanceはclean Git revisionで実行してください")
        target = self._environment_probe()
        storage_preflight = (
            self._storage_preflight(profile, input_folder)
            if state is None or not _runs_completed(state)
            else _mapping(state.get("storage_preflight"), "storage_preflight")
        )
        _validate_full_scan_capacity(
            suite,
            cold_configuration,
            target,
            suite_descriptor,
        )
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
            effective_configuration_digest,
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
                "configuration": configuration_summary,
                "models": resolved_models.provenance(),
                "storage_preflight": storage_preflight,
                "phases": {},
            }
            if suite == "full":
                state["comparison_runs"] = {}
            write_atomic_json(state_path, state)
        else:
            _validate_state_identity(state, identity)
            if not _runs_completed(state):
                _refresh_execution_context(
                    state,
                    identity=identity,
                    source_revision={"commit": commit, "dirty": False},
                    target=target,
                    configuration=configuration_summary,
                    models=resolved_models.provenance(),
                    storage_preflight=storage_preflight,
                )
                write_atomic_json(state_path, state)

        execution_context = _attempt_execution_context(
            identity=identity,
            source_revision={"commit": commit, "dirty": False},
            target=target,
            configuration=configuration_summary,
            models=resolved_models.provenance(),
        )
        _reconcile_full_comparison_context(
            suite=suite,
            state=state,
            execution_context=execution_context,
            suite_root=suite_root,
            cache_folder=cold_configuration.processing_cache_folder,
            state_path=state_path,
        )
        _reconcile_release_evidence_context(
            suite=suite,
            state=state,
            execution_context=execution_context,
            execution_steps=execution_steps,
            suite_root=suite_root,
            cache_folder=cold_configuration.processing_cache_folder,
            state_path=state_path,
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
                load_completed_run_report(
                    configuration=step.configuration,
                    run_record=cast(dict[str, object], existing),
                )
                continue
            _remove_invalid_attempt_output(step.configuration.output_folder)
            attempt_id = uuid4().hex
            attempt_started_at_epoch = time.time()
            state[step.active_state_key] = step.name
            state[_active_attempt_started_key(step)] = attempt_started_at_epoch
            state[_active_attempt_id_key(step)] = attempt_id
            state[_active_attempt_context_key(step)] = execution_context
            write_atomic_json(state_path, state)
            attempt_journal.start(
                attempt_id=attempt_id,
                step_kind=step.kind,
                step_name=step.name,
                started_at_epoch_seconds=attempt_started_at_epoch,
                execution_context=execution_context,
            )
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
                recovered = attempt_journal.recover(
                    attempt_id=attempt_id,
                    step_kind=step.kind,
                    step_name=step.name,
                    processing_cache_folder=(
                        step.configuration.processing_cache_folder
                    ),
                    video_identity_cache_folder=(
                        step.configuration.durable_video_identity_cache_folder
                    ),
                )
                attempt_record = build_incomplete_interrupt_attempt(
                    time.monotonic() - attempt_started_at,
                    None if recovered is None else recovered[0],
                )
                report = None
                selection = None
            except Exception:
                prior_attempts = _run_attempts(state, step)
                recovered = attempt_journal.recover(
                    attempt_id=attempt_id,
                    step_kind=step.kind,
                    step_name=step.name,
                    processing_cache_folder=(
                        step.configuration.processing_cache_folder
                    ),
                    video_identity_cache_folder=(
                        step.configuration.durable_video_identity_cache_folder
                    ),
                )
                abandoned_attempt = build_incomplete_interrupt_attempt(
                    time.monotonic() - attempt_started_at,
                    None if recovered is None else recovered[0],
                )
                abandoned_attempt["failure_reason"] = "process_abandoned"
                abandoned_attempt["failure_exit_code"] = 1
                abandoned_attempt["attempt_id"] = attempt_id
                abandoned_attempt["execution_context"] = (
                    execution_context if recovered is None else recovered[1]
                )
                _store_run_attempts(
                    state,
                    step,
                    (*prior_attempts, abandoned_attempt),
                )
                state.pop(step.active_state_key, None)
                state.pop(_active_attempt_started_key(step), None)
                state.pop(_active_attempt_id_key(step), None)
                state.pop(_active_attempt_context_key(step), None)
                state["last_failure"] = {
                    **step.failure_context,
                    "exit_code": 1,
                    "reason": "run_measurement_incomplete",
                }
                write_atomic_json(state_path, state)
                attempt_journal.clear()
                return 1
            attempt_record["attempt_id"] = attempt_id
            attempt_record["execution_context"] = execution_context
            validate_run_measurements(attempt_record)
            prior_attempts = _run_attempts(state, step)
            state.pop(step.active_state_key, None)
            state.pop(_active_attempt_started_key(step), None)
            state.pop(_active_attempt_id_key(step), None)
            state.pop(_active_attempt_context_key(step), None)
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
                attempt_journal.clear()
                return exit_code
            if attempt_record.get("operation_status") != "completed":
                raise ValueError("成功runの計測記録がcompletedではありません")
            runs[step.name] = aggregate_run_attempts((*prior_attempts, attempt_record))
            state[step.records_state_key] = runs
            _clear_run_attempts(state, step)
            state.pop("last_failure", None)
            write_atomic_json(state_path, state)
            attempt_journal.clear()
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
        if record["status"] == "passed" and not _cleanup_release_work(
            suite,
            suite_root,
            state,
        ):
            return 1
        state_path = suite_root / "acceptance-state.json"
        state["acceptance_status"] = "finalizing"
        state["last_failure"] = {
            "phase": "acceptance_finalization",
            "exit_code": 1,
            "reason": "acceptance_finalization_in_progress",
        }
        write_atomic_json(state_path, state)
        try:
            _remove_directory_strict(
                suite_root / "baseline",
                "Acceptance baseline",
            )
            if record["status"] == "passed":
                write_normalized_baseline(record, suite_root / "baseline")
            write_atomic_json(suite_root / "acceptance.json", record)
            state["acceptance_status"] = record["status"]
            state.pop("last_failure", None)
            write_atomic_json(state_path, state)
        except BaseException as error:
            state["acceptance_status"] = "failed"
            state["last_failure"] = {
                "phase": "acceptance_finalization",
                "exit_code": 130 if isinstance(error, KeyboardInterrupt) else 1,
                "reason": "acceptance_finalization_failed",
            }
            with suppress(Exception):
                write_atomic_json(state_path, state)
            raise
        if record["status"] == "passed":
            return 0
        if record["status"] == "pending_human_review":
            return 3
        return 1


def _reset_acceptance_run_suffix(
    *,
    reset_run: AcceptanceRunReset,
    state: dict[str, object],
    execution_steps: tuple[AcceptanceExecutionStep, ...],
    suite_root: Path,
    processing_cache_folder: Path,
    state_path: Path,
) -> None:
    """指定runと依存する後続runだけをstateとartifactから破棄する。"""
    reset_names = {
        "parallelism-baseline": {"fixed3", "cold", "warm"},
        "fresh-processing": {"cold", "warm"},
        "cache-reuse": {"warm"},
    }[reset_run]
    reset_steps = tuple(step for step in execution_steps if step.name in reset_names)
    for step in reset_steps:
        _remove_directory_strict(
            step.configuration.output_folder,
            f"{reset_run} Acceptance output",
        )
    if "cold" in reset_names:
        _remove_directory_strict(
            processing_cache_folder,
            f"{reset_run} processing cache",
        )
        _remove_file_strict(
            suite_root / "review-worksheet.json",
            "Acceptance review worksheet",
        )
    _remove_file_strict(
        suite_root / "acceptance.json",
        "Acceptance record",
    )
    _remove_directory_strict(
        suite_root / "baseline",
        "Acceptance baseline",
    )

    for step in reset_steps:
        records = state.get(step.records_state_key)
        if isinstance(records, dict):
            records.pop(step.name, None)
        attempts = state.get(step.attempts_state_key)
        if isinstance(attempts, dict):
            attempts.pop(step.name, None)
        if state.get(step.active_state_key) == step.name:
            state.pop(step.active_state_key, None)
            state.pop(_active_attempt_started_key(step), None)
            state.pop(_active_attempt_id_key(step), None)
            state.pop(_active_attempt_context_key(step), None)
    if reset_run == "parallelism-baseline":
        state.pop("fixed3_cache_released", None)
    if "cold" in reset_names:
        state.pop("worksheet_ready", None)
        state.pop("review_candidate_digest", None)
        state.pop("video_set", None)
    state.pop("acceptance_status", None)
    state.pop("last_failure", None)
    write_atomic_json(state_path, state)


def _validate_cache_reuse_reset_prerequisites(
    state: dict[str, object] | None,
    processing_cache_folder: Path,
) -> None:
    """本処理の完了recordと実cacheがある場合だけ再利用resetを許可する。"""
    phases = state.get("phases") if state is not None else None
    fresh_processing = phases.get("cold") if isinstance(phases, dict) else None
    if (
        not isinstance(fresh_processing, dict)
        or fresh_processing.get("operation_status") != "completed"
        or processing_cache_folder.is_symlink()
        or not processing_cache_folder.is_dir()
    ):
        raise ValueError(
            "cache-reuse resetに必要な本処理cacheがありません。"
            "--reset-run fresh-processingで本処理から再測定してください"
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


def _remove_file_strict(path: Path, label: str) -> None:
    """suite-owned regular fileだけを削除し外部参照を辿らない。"""
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label}が通常fileではありません")
    try:
        path.unlink()
    except OSError:
        raise ValueError(f"{label}を削除できません") from None
    if path.exists() or path.is_symlink():
        raise ValueError(f"{label}を完全に削除できません")


def _remove_invalid_attempt_output(output_folder: Path) -> None:
    """完成済みCanonical outputを保持し、不完全なsuite-owned outputだけ除く。"""
    if not output_folder.exists() and not output_folder.is_symlink():
        return
    if output_folder.is_symlink() or not output_folder.is_dir():
        raise ValueError("Acceptance Output Folderがdirectoryではありません")
    try:
        load_validated_canonical_selection_report(output_folder)
    except ValueError:
        _remove_directory_strict(output_folder, "不完全なAcceptance Output Folder")


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
    identity_cache_folder: Path,
) -> EffectiveConfiguration:
    return replace(
        resolve_effective_configuration(
            video_input_folder=input_folder,
            output_folder=output_folder,
            config_path=profile.configuration_path,
        ),
        video_identity_cache_folder=identity_cache_folder,
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


def _validate_full_scan_capacity(
    suite: str,
    configuration: EffectiveConfiguration,
    target: Mapping[str, object],
    descriptor: Mapping[str, object],
) -> None:
    if suite != "full":
        return
    if configuration.video_scan_workers != "auto":
        raise ValueError("Full acceptanceのVideo Scan workersにはautoが必要です")
    scenario_count = _positive_integer(
        descriptor.get("scenario_count"),
        "Full suite scenario count",
    )
    logical_cpu_count = _positive_integer(
        target.get("logical_cpu_count"),
        "Target logical CPU count",
    )
    controller = AdaptiveVideoScanController(
        video_count=scenario_count,
        configured_workers=configuration.video_scan_workers,
        auto_max_workers=configuration.video_scan_auto_max_workers,
        decode_backend=configuration.decode_backend,
        logical_cpu_count=logical_cpu_count,
        initial_resource_sample=None,
    )
    if controller.maximum_reachable_workers <= 3:
        raise ValueError(
            "Full acceptanceのVideo Scanでは実際に4 worker以上へ"
            "到達できるVideo数が必要です"
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


def _reconcile_release_evidence_context(
    *,
    suite: str,
    state: dict[str, object],
    execution_context: Mapping[str, object],
    execution_steps: tuple[AcceptanceExecutionStep, ...],
    suite_root: Path,
    cache_folder: Path,
    state_path: Path,
) -> None:
    """releaseの本処理とcache再利用を同じEvidence Contextへ揃える。"""
    if suite != "release" or _runs_completed(state):
        return
    phases = _mapping(state.get("phases"), "phases")
    fresh_processing = phases.get("cold")
    if (
        not isinstance(fresh_processing, dict)
        or fresh_processing.get("operation_status") != "completed"
    ):
        return
    fresh_processing_matches = acceptance_run_matches_evidence_context(
        fresh_processing,
        execution_context,
    )
    reuse_attempts = state.get("phase_attempts")
    raw_reuse_attempts = (
        reuse_attempts.get("warm") if isinstance(reuse_attempts, dict) else None
    )
    reuse_started = isinstance(phases.get("warm"), dict) or (
        isinstance(raw_reuse_attempts, list) and bool(raw_reuse_attempts)
    )
    if reuse_started:
        reuse_matches = (
            isinstance(raw_reuse_attempts, list)
            and bool(raw_reuse_attempts)
            and acceptance_run_matches_evidence_context(
                {"attempts": raw_reuse_attempts},
                execution_context,
            )
        )
        if not fresh_processing_matches or not reuse_matches:
            raise ValueError(
                "Cache reuse開始後にAcceptance Evidence Contextが変更されました。"
                "--reset-run fresh-processingで本処理から再測定してください"
            )
        return
    if fresh_processing_matches:
        return
    _reset_acceptance_run_suffix(
        reset_run="fresh-processing",
        state=state,
        execution_steps=execution_steps,
        suite_root=suite_root,
        processing_cache_folder=cache_folder,
        state_path=state_path,
    )
    remeasurement_count = state.get("fresh_processing_remeasurement_count", 0)
    if not isinstance(remeasurement_count, int) or isinstance(
        remeasurement_count,
        bool,
    ):
        raise ValueError("Fresh processing再測定回数が不正です")
    state["fresh_processing_remeasurement_count"] = remeasurement_count + 1
    write_atomic_json(state_path, state)


def _reconcile_full_comparison_context(
    *,
    suite: str,
    state: dict[str, object],
    execution_context: Mapping[str, object],
    suite_root: Path,
    cache_folder: Path,
    state_path: Path,
) -> None:
    """fixed3とauto coldを同じVideo Scan Comparison Contextへ揃える。"""
    if suite != "full" or _runs_completed(state):
        return
    comparison_runs = _mapping(state.get("comparison_runs"), "comparison_runs")
    fixed_three = comparison_runs.get("fixed3")
    if (
        not isinstance(fixed_three, dict)
        or fixed_three.get("operation_status") != "completed"
    ):
        return
    fixed_three_matches = video_scan_run_matches_comparison_context(
        fixed_three,
        execution_context,
    )
    cold_started = _auto_cold_started(state)
    if cold_started:
        if not fixed_three_matches or not _auto_cold_matches_comparison_context(
            state,
            execution_context,
        ):
            raise ValueError(
                "Fresh Processing開始後にVideo Scan Comparison Contextが"
                "変更されました。--reset-run parallelism-baselineで"
                "並列基準から再測定してください"
            )
        return
    if fixed_three_matches:
        return
    _remove_directory_strict(
        suite_root / "outputs" / "fixed3",
        "旧contextのFixed3 comparison output",
    )
    _remove_directory_strict(
        cache_folder,
        "旧contextのFixed3 comparison cache",
    )
    comparison_runs.pop("fixed3", None)
    state["comparison_runs"] = comparison_runs
    comparison_attempts = state.get("comparison_run_attempts")
    if isinstance(comparison_attempts, dict):
        comparison_attempts.pop("fixed3", None)
    state.pop("fixed3_cache_released", None)
    remeasurement_count = state.get("fixed3_remeasurement_count", 0)
    if not isinstance(remeasurement_count, int) or isinstance(
        remeasurement_count,
        bool,
    ):
        raise ValueError("Fixed3 comparison再測定回数が不正です")
    state["fixed3_remeasurement_count"] = remeasurement_count + 1
    write_atomic_json(state_path, state)


def _auto_cold_started(state: Mapping[str, object]) -> bool:
    phases = state.get("phases")
    if isinstance(phases, dict) and isinstance(phases.get("cold"), dict):
        return True
    attempts = state.get("phase_attempts")
    return (
        isinstance(attempts, dict)
        and isinstance(attempts.get("cold"), list)
        and bool(attempts["cold"])
    )


def _auto_cold_matches_comparison_context(
    state: Mapping[str, object],
    execution_context: Mapping[str, object],
) -> bool:
    phases = state.get("phases")
    if (
        isinstance(phases, dict)
        and isinstance(cold := phases.get("cold"), dict)
        and not video_scan_run_matches_comparison_context(
            cold,
            execution_context,
        )
    ):
        return False
    attempts = state.get("phase_attempts")
    if (
        isinstance(attempts, dict)
        and isinstance(cold_attempts := attempts.get("cold"), list)
        and cold_attempts
    ):
        if any(not isinstance(attempt, dict) for attempt in cold_attempts):
            return False
        return video_scan_run_matches_comparison_context(
            {"attempts": cold_attempts},
            execution_context,
        )
    return True


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


def _effective_configuration_digest(
    summary: Mapping[str, object],
) -> str:
    payload = json.dumps(
        summary,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(
        b"game-screen-pick/effective-configuration@1\0" + payload
    ).hexdigest()


def _identity(
    suite: str,
    profile: AcceptanceProfile,
    configuration_digest: str,
    configuration: EffectiveConfiguration,
    effective_configuration_digest: str,
    descriptor: Mapping[str, object],
    models: ResolvedModels,
    commit: str,
) -> dict[str, object]:
    descriptor_fingerprint = descriptor.get(
        "suite_fingerprint",
        descriptor.get("source_snapshot_fingerprint"),
    )
    materialization_source_snapshot = descriptor.get("source_snapshot_fingerprint")
    if not isinstance(descriptor_fingerprint, str):
        raise ValueError("Suite materialization fingerprintがありません")
    if not _is_sha256(materialization_source_snapshot):
        raise ValueError("Suite source snapshot fingerprintがありません")
    models_json = json.dumps(
        models.semantic_input(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {
        "suite": suite,
        "profile_digest": profile.profile_digest,
        "configuration_digest": configuration_digest,
        "effective_configuration_digest": effective_configuration_digest,
        "ollama_endpoint_identity": _ollama_endpoint_identity(
            configuration.ollama_host
        ),
        "suite_fingerprint": descriptor_fingerprint,
        "materialization_source_snapshot_fingerprint": (
            materialization_source_snapshot
        ),
        "model_identity_digest": hashlib.sha256(models_json).hexdigest(),
        "commit": commit,
    }


def _validate_profile_files(profile: AcceptanceProfile) -> None:
    if not profile.input_root.is_dir():
        raise ValueError("Acceptance input rootが存在しません")
    if not profile.configuration_path.is_file():
        raise ValueError("Acceptance configurationが存在しません")
    profile.artifact_root.mkdir(parents=True, exist_ok=True)


def _validate_suite_source_paths(
    profile_path: Path,
    profile: AcceptanceProfile,
    suite_root: Path,
) -> None:
    protected_paths = (
        ("Acceptance input root", profile.input_root),
        ("Acceptance configuration", profile.configuration_path),
        ("Acceptance profile", profile_path),
    )
    for label, path in protected_paths:
        if _path_is_within(path, suite_root):
            raise ValueError(f"{label}はsuite削除対象directory内に置けません")


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path_variants = (path.absolute(), path.resolve(strict=False))
        parent_variants = (parent.absolute(), parent.resolve(strict=False))
    except (OSError, RuntimeError):
        return True
    for candidate in path_variants:
        for root in parent_variants:
            try:
                candidate.relative_to(root)
            except ValueError:
                continue
            return True
    return False


def _validate_state_identity(
    state: Mapping[str, object],
    identity: Mapping[str, object],
) -> None:
    stable_keys = (
        "suite",
        "profile_digest",
        "suite_fingerprint",
        "materialization_source_snapshot_fingerprint",
    )
    if state.get("schema") != _STATE_SCHEMA or any(
        state.get(key) != identity.get(key) for key in stable_keys
    ):
        raise ValueError("Acceptance stateが現在のsuite identityと一致しません")


def _validate_completed_state_identity(
    state: Mapping[str, object],
    suite: str,
    profile: AcceptanceProfile,
    current_source_snapshot: str,
) -> None:
    """materialization削除後も検証できる完了済みsuite identityを確認する。"""
    suite_fingerprint = state.get("suite_fingerprint")
    if not _is_sha256(suite_fingerprint):
        raise ValueError("Acceptance stateのsuite fingerprintが不正です")
    recorded_source_snapshot = state.get("materialization_source_snapshot_fingerprint")
    if (
        not _is_sha256(recorded_source_snapshot)
        or recorded_source_snapshot != current_source_snapshot
    ):
        raise ValueError("Acceptance stateのsource snapshotが変更されています")
    _validate_state_identity(
        state,
        {
            "suite": suite,
            "profile_digest": profile.profile_digest,
            "suite_fingerprint": suite_fingerprint,
            "materialization_source_snapshot_fingerprint": (recorded_source_snapshot),
        },
    )


def _refresh_execution_context(
    state: dict[str, object],
    *,
    identity: Mapping[str, object],
    source_revision: Mapping[str, object],
    target: Mapping[str, object],
    configuration: Mapping[str, object],
    models: Mapping[str, object],
    storage_preflight: Mapping[str, object],
) -> None:
    """未完了suiteの実行依存を現在attemptへ更新する。"""
    mutable_identity_keys = (
        "configuration_digest",
        "effective_configuration_digest",
        "ollama_endpoint_identity",
        "model_identity_digest",
        "commit",
    )
    previous = {key: state.get(key) for key in mutable_identity_keys}
    current = {key: identity.get(key) for key in mutable_identity_keys}
    if previous != current:
        history = state.get("execution_context_changes")
        changes = history if isinstance(history, list) else []
        changes.append(
            {
                "previous": previous,
                "current": current,
            }
        )
        state["execution_context_changes"] = changes
    state.update(current)
    state["source_revision"] = dict(source_revision)
    state["target"] = dict(target)
    state["configuration"] = dict(configuration)
    state["models"] = {
        key: dict(value) if isinstance(value, dict) else value
        for key, value in models.items()
    }
    state["storage_preflight"] = dict(storage_preflight)


def _attempt_execution_context(
    *,
    identity: Mapping[str, object],
    source_revision: Mapping[str, object],
    target: Mapping[str, object],
    configuration: Mapping[str, object],
    models: Mapping[str, object],
) -> dict[str, object]:
    """一つのattemptでfreezeされたprivacy-safe実行依存を返す。"""
    mutable_identity_keys = (
        "configuration_digest",
        "effective_configuration_digest",
        "ollama_endpoint_identity",
        "model_identity_digest",
        "commit",
    )
    return {
        "identity": {key: identity.get(key) for key in mutable_identity_keys},
        "source_revision": dict(source_revision),
        "target": dict(target),
        "configuration": dict(configuration),
        "models": {
            key: dict(value) if isinstance(value, dict) else value
            for key, value in models.items()
        },
    }


def _recover_abandoned_attempt(
    state: dict[str, object],
    steps: tuple[AcceptanceExecutionStep, ...],
    state_path: Path,
    attempt_journal: AcceptanceAttemptJournal,
) -> None:
    """process終了で残ったactive markerを保守的なattemptへ閉じる。"""
    active_markers = {
        key: state[key] for key in _ACTIVE_RUN_STATE_KEYS if state.get(key) is not None
    }
    if not active_markers:
        return
    if len(active_markers) != 1:
        raise ValueError("複数のAcceptance Runが同時にactiveです")
    active_key, active_name = next(iter(active_markers.items()))
    active_steps = tuple(
        step
        for step in steps
        if step.active_state_key == active_key and step.name == active_name
    )
    if len(active_steps) != 1:
        raise ValueError("Acceptance active runが現在のexecution planと一致しません")
    step = active_steps[0]
    started_at = state.get(_active_attempt_started_key(step))
    duration_seconds = (
        max(0.0, time.time() - float(started_at))
        if isinstance(started_at, int | float) and not isinstance(started_at, bool)
        else 0.0
    )
    attempt_id_value = state.get(_active_attempt_id_key(step))
    attempt_id = (
        attempt_id_value
        if isinstance(attempt_id_value, str) and attempt_id_value
        else f"legacy-{step.kind}-{step.name}"
    )
    context_value = state.get(_active_attempt_context_key(step))
    execution_context = (
        dict(context_value)
        if isinstance(context_value, dict)
        else _attempt_execution_context(
            identity=state,
            source_revision=_mapping(
                state.get("source_revision"),
                "source_revision",
            ),
            target=_mapping(state.get("target"), "target"),
            configuration=_mapping(
                state.get("configuration"),
                "configuration",
            ),
            models=_mapping(state.get("models"), "models"),
        )
    )
    recovered = (
        attempt_journal.recover(
            attempt_id=attempt_id,
            step_kind=step.kind,
            step_name=step.name,
            processing_cache_folder=step.configuration.processing_cache_folder,
            video_identity_cache_folder=(
                step.configuration.durable_video_identity_cache_folder
            ),
        )
        if attempt_journal.exists and not attempt_id.startswith("legacy-")
        else None
    )
    recovered_metrics = None if recovered is None else recovered[0]
    if recovered is not None:
        execution_context = recovered[1]
    attempt = build_incomplete_interrupt_attempt(
        duration_seconds,
        recovered_metrics,
    )
    attempt["failure_reason"] = "process_abandoned"
    attempt["failure_exit_code"] = 1
    attempt["attempt_id"] = attempt_id
    attempt["execution_context"] = execution_context
    _store_run_attempts(
        state,
        step,
        (*_run_attempts(state, step), attempt),
    )
    state.pop(step.active_state_key, None)
    state.pop(_active_attempt_started_key(step), None)
    state.pop(_active_attempt_id_key(step), None)
    state.pop(_active_attempt_context_key(step), None)
    state["last_failure"] = {
        **step.failure_context,
        "exit_code": 1,
        "reason": "process_abandoned",
    }
    write_atomic_json(state_path, state)
    attempt_journal.clear()


def _active_attempt_started_key(step: AcceptanceExecutionStep) -> str:
    return f"{step.active_state_key}_started_at_epoch_seconds"


def _active_attempt_id_key(step: AcceptanceExecutionStep) -> str:
    return f"{step.active_state_key}_attempt_id"


def _active_attempt_context_key(step: AcceptanceExecutionStep) -> str:
    return f"{step.active_state_key}_execution_context"


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


def _positive_integer(value: object, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{label}が正の整数ではありません")
    return value


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
