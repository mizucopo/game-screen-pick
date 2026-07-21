"""cold/warm resume、record、human gateを所有するtarget suite runner。"""

import hashlib
import json
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

from ..configuration.resolve_effective_configuration import (
    resolve_effective_configuration,
)
from ..model_runtime.model_lifecycle_runtime import ModelLifecycleRuntime
from ..models.effective_configuration import EffectiveConfiguration
from ..models.resolved_models import ResolvedModels
from .acceptance_profile import AcceptanceProfile
from .acceptance_record import (
    build_acceptance_record,
    validate_acceptance_record_privacy,
    write_normalized_baseline,
)
from .atomic_json import read_json_object, write_atomic_json
from .execute_acceptance_phase import (
    PhaseExecutionResult,
    execute_acceptance_phase,
    load_completed_phase_evidence,
    public_phase_record,
)
from .full_suite_materializer import FullSuiteMaterializer
from .human_review import ensure_review_worksheet, evaluate_human_review
from .load_acceptance_profile import load_acceptance_profile
from .release_suite_materializer import ReleaseSuiteMaterializer
from .target_environment import probe_source_revision, probe_target_environment

EnvironmentProbe = Callable[[], dict[str, object]]
RevisionProbe = Callable[[Path], tuple[str, bool]]
ModelResolver = Callable[[EffectiveConfiguration], ResolvedModels]
PhaseExecutor = Callable[
    [str, EffectiveConfiguration, ResolvedModels, Path],
    PhaseExecutionResult,
]
SuiteMaterializer = Callable[
    [AcceptanceProfile, Path],
    tuple[Path, dict[str, object]],
]

_STATE_SCHEMA = "game-screen-pick/target-acceptance-state@1.0.0"


class TargetSuiteRunner:
    """一回のsuiteでcold→exact warm→human gateをdurableに進める。"""

    def __init__(
        self,
        *,
        environment_probe: EnvironmentProbe = probe_target_environment,
        revision_probe: RevisionProbe = probe_source_revision,
        model_resolver: ModelResolver | None = None,
        phase_executor: PhaseExecutor | None = None,
        release_materializer: SuiteMaterializer | None = None,
        full_materializer: SuiteMaterializer | None = None,
    ) -> None:
        self._environment_probe = environment_probe
        self._revision_probe = revision_probe
        self._model_resolver = model_resolver or ModelLifecycleRuntime().resolve_models
        self._phase_executor = phase_executor or _execute_phase
        self._release_materializer = (
            release_materializer or ReleaseSuiteMaterializer().materialize
        )
        self._full_materializer = (
            full_materializer or FullSuiteMaterializer().materialize
        )

    def run(
        self,
        *,
        profile_path: Path,
        suite: str,
        reset_suite: bool = False,
        human_review_path: Path | None = None,
    ) -> int:
        """suiteを未完了phaseから進めacceptance exit codeを返す。"""
        if suite not in {"release", "full"}:
            raise ValueError("--suiteにはreleaseまたはfullが必要です")
        profile = load_acceptance_profile(profile_path)
        _validate_profile_files(profile)
        suite_root = profile.artifact_root / "target-acceptance" / suite
        if reset_suite:
            shutil.rmtree(suite_root, ignore_errors=True)
        state_path = suite_root / "acceptance-state.json"
        state = read_json_object(state_path)
        configuration_digest = _content_digest(profile.configuration_path)
        commit, dirty = self._revision_probe(Path.cwd())
        if dirty:
            raise ValueError("Target acceptanceはclean Git revisionで実行してください")
        if state is not None and state.get("active_phase") is not None:
            raise ValueError("前回phaseの計測が未確定です。--reset-suiteが必要です")

        target = self._environment_probe()
        input_folder, suite_descriptor = self._materialize(
            profile,
            suite,
            suite_root,
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
        try:
            resolved_models = self._model_resolver(cold_configuration)
        except KeyboardInterrupt:
            return 130
        except Exception:
            return 1
        identity = _identity(
            suite,
            profile,
            configuration_digest,
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
                "configuration": _configuration_summary(cold_configuration),
                "models": resolved_models.provenance(),
                "phases": {},
            }
            write_atomic_json(state_path, state)
        else:
            _validate_state_identity(state, identity)

        if _phases_completed(state) and state.get("worksheet_ready") is True:
            return self._finalize(
                profile,
                suite_root,
                state,
                human_review_path=human_review_path,
            )

        phases = _mapping(state.get("phases"), "phases")
        cold_report: dict[str, object] | None = None
        cold_selection: dict[str, object] | None = None
        for phase, configuration in (
            ("cold", cold_configuration),
            ("warm", warm_configuration),
        ):
            existing = phases.get(phase)
            if (
                isinstance(existing, dict)
                and existing.get("operation_status") == "completed"
            ):
                continue
            shutil.rmtree(configuration.output_folder, ignore_errors=True)
            state["active_phase"] = phase
            write_atomic_json(state_path, state)
            try:
                exit_code, record, report, selection = self._phase_executor(
                    phase,
                    configuration,
                    resolved_models,
                    suite_root,
                )
            except KeyboardInterrupt:
                state["last_failure"] = {
                    "phase": phase,
                    "exit_code": 130,
                    "reason": "phase_measurement_incomplete",
                }
                write_atomic_json(state_path, state)
                return 130
            except Exception:
                state["last_failure"] = {
                    "phase": phase,
                    "exit_code": 1,
                    "reason": "phase_measurement_incomplete",
                }
                write_atomic_json(state_path, state)
                return 1
            _validate_phase_measurements(record)
            prior_attempts = _phase_attempts(state, phase)
            state.pop("active_phase", None)
            if exit_code != 0:
                _store_phase_attempts(state, phase, (*prior_attempts, record))
                state["last_failure"] = {
                    "phase": phase,
                    "exit_code": exit_code,
                    "reason": record.get("failure_reason", "operation_failed"),
                }
                write_atomic_json(state_path, state)
                return exit_code
            if record.get("operation_status") != "completed":
                raise ValueError("成功phaseの計測記録がcompletedではありません")
            phases[phase] = _aggregate_phase_attempts((*prior_attempts, record))
            state["phases"] = phases
            _clear_phase_attempts(state, phase)
            state.pop("last_failure", None)
            write_atomic_json(state_path, state)
            if phase == "cold":
                cold_report = report
                cold_selection = selection

        if cold_report is None or cold_selection is None:
            cold_phase = _mapping(phases.get("cold"), "cold phase")
            cold_report, cold_selection = load_completed_phase_evidence(
                configuration=cold_configuration,
                phase_record=cold_phase,
            )
        cold_video_set = _mapping(
            _mapping(phases.get("cold"), "cold phase").get("video_set"),
            "video_set",
        )
        state["video_set"] = cold_video_set
        ensure_review_worksheet(
            suite_root / "review-worksheet.json",
            suite=suite,
            suite_fingerprint=_string(state.get("suite_fingerprint")),
            canonical_report=cold_report,
            selection_artifact=cold_selection,
        )
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
        )
        phases = _mapping(state.get("phases"), "phases")
        cold = _mapping(phases.get("cold"), "cold phase")
        warm = _mapping(phases.get("warm"), "warm phase")
        revision = _mapping(state.get("source_revision"), "source_revision")
        record = build_acceptance_record(
            suite=suite,
            commit=_string(revision.get("commit")),
            dirty=_boolean(revision.get("dirty")),
            target=_mapping(state.get("target"), "target"),
            configuration=_mapping(state.get("configuration"), "configuration"),
            models=_mapping(state.get("models"), "models"),
            video_set=_mapping(state.get("video_set"), "video_set"),
            cold=public_phase_record(cold),
            warm=public_phase_record(warm),
            human_quality=human_quality,
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
            write_atomic_json(suite_root / "acceptance-state.json", state)
            if suite == "release":
                shutil.rmtree(suite_root / "work", ignore_errors=True)
            return 1
        write_atomic_json(suite_root / "acceptance.json", record)
        try:
            state["acceptance_status"] = record["status"]
            state.pop("last_failure", None)
            write_atomic_json(suite_root / "acceptance-state.json", state)
            if record["status"] == "passed":
                write_normalized_baseline(record, suite_root / "baseline")
        finally:
            if suite == "release":
                shutil.rmtree(suite_root / "work", ignore_errors=True)
        return (
            0
            if record["status"] == "passed"
            else 3
            if record["status"] == "pending_human_review"
            else 1
        )


def _execute_phase(
    phase: str,
    configuration: EffectiveConfiguration,
    resolved_models: ResolvedModels,
    suite_root: Path,
) -> PhaseExecutionResult:
    return execute_acceptance_phase(
        phase=phase,
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


def _configuration_summary(
    configuration: EffectiveConfiguration,
) -> dict[str, object]:
    return {
        "config_version": configuration.config_version,
        "recursive": configuration.recursive,
        "image_count": configuration.image_count,
        "spoiler_sensitivity": configuration.spoiler_sensitivity,
        "similarity_threshold": configuration.similarity_threshold,
        "decode_backend": configuration.decode_backend,
        "candidate_density_per_minute": (configuration.candidate_density_per_minute),
        "language": configuration.language,
        "models_auto_upgrade": configuration.models_auto_upgrade,
        "speech_to_text_device": configuration.speech_to_text_device,
        "speech_to_text_compute_type": configuration.speech_to_text_compute_type,
    }


def _identity(
    suite: str,
    profile: AcceptanceProfile,
    configuration_digest: str,
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
        models.provenance(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {
        "suite": suite,
        "profile_digest": profile.profile_digest,
        "configuration_digest": configuration_digest,
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


def _phases_completed(state: Mapping[str, object]) -> bool:
    phases = state.get("phases")
    if not isinstance(phases, dict):
        return False
    return all(
        isinstance(phases.get(phase), dict)
        and phases[phase].get("operation_status") == "completed"
        for phase in ("cold", "warm")
    )


_SUMMED_INTEGER_METRICS = (
    "cache_hit_count",
    "cache_miss_count",
    "reuse_count",
    "unexpected_recompute_count",
    "disk_sample_count",
    "gpu_sample_count",
    "gpu_sample_error_count",
)
_MAXIMUM_INTEGER_METRICS = (
    "persistent_cache_bytes",
    "peak_additional_bytes",
    "system_global_gpu_peak_mib",
    "ollama_global_gpu_peak_mib",
    "stt_global_gpu_peak_mib",
    "ollama_model_size_vram_bytes",
)
_BASELINE_INTEGER_METRICS = (
    "process_gpu_baseline_mib",
    "system_gpu_baseline_mib",
)


def _phase_attempts(
    state: Mapping[str, object],
    phase: str,
) -> tuple[dict[str, object], ...]:
    attempts = state.get("phase_attempts")
    if attempts is None:
        return ()
    attempts_by_phase = _mapping(attempts, "phase attempts")
    values = attempts_by_phase.get(phase, [])
    if not isinstance(values, list) or any(
        not isinstance(value, dict) for value in values
    ):
        raise ValueError("Acceptance state phase attemptsが不正です")
    return tuple(cast(dict[str, object], value) for value in values)


def _store_phase_attempts(
    state: dict[str, object],
    phase: str,
    attempts: tuple[Mapping[str, object], ...],
) -> None:
    existing = state.get("phase_attempts")
    attempts_by_phase = {} if existing is None else _mapping(existing, "phase attempts")
    attempts_by_phase[phase] = [dict(attempt) for attempt in attempts]
    state["phase_attempts"] = attempts_by_phase


def _clear_phase_attempts(state: dict[str, object], phase: str) -> None:
    existing = state.get("phase_attempts")
    if existing is None:
        return
    attempts_by_phase = _mapping(existing, "phase attempts")
    attempts_by_phase.pop(phase, None)
    if attempts_by_phase:
        state["phase_attempts"] = attempts_by_phase
    else:
        state.pop("phase_attempts", None)


def _validate_phase_measurements(record: Mapping[str, object]) -> None:
    _measurement_number(record, "duration_seconds")
    for key in (
        *_SUMMED_INTEGER_METRICS,
        *_MAXIMUM_INTEGER_METRICS,
        *_BASELINE_INTEGER_METRICS,
    ):
        _measurement_integer(record, key)
    _measurement_numeric_mapping(record, "stage_durations_seconds")
    _measurement_integer_mapping(record, "completed_stage_counts")
    if not isinstance(record.get("resource_sampling_complete"), bool):
        raise ValueError("Acceptance phase metric resource_sampling_completeが不正です")


def _aggregate_phase_attempts(
    records: tuple[Mapping[str, object], ...],
) -> dict[str, object]:
    if not records:
        raise ValueError("Acceptance phase attemptがありません")
    for record in records:
        _validate_phase_measurements(record)
    aggregate = dict(records[-1])
    aggregate["attempt_count"] = len(records)
    aggregate["duration_seconds"] = sum(
        _measurement_number(record, "duration_seconds") for record in records
    )
    for key in _SUMMED_INTEGER_METRICS:
        aggregate[key] = sum(_measurement_integer(record, key) for record in records)
    for key in _MAXIMUM_INTEGER_METRICS:
        aggregate[key] = max(_measurement_integer(record, key) for record in records)
    for key in _BASELINE_INTEGER_METRICS:
        aggregate[key] = _measurement_integer(records[0], key)
    aggregate["stage_durations_seconds"] = _sum_numeric_mappings(
        records,
        "stage_durations_seconds",
    )
    aggregate["completed_stage_counts"] = _sum_integer_mappings(
        records,
        "completed_stage_counts",
    )
    aggregate["resource_sampling_complete"] = all(
        record["resource_sampling_complete"] is True for record in records
    )
    aggregate.pop("failure_reason", None)
    aggregate.pop("failure_exit_code", None)
    return aggregate


def _sum_numeric_mappings(
    records: tuple[Mapping[str, object], ...],
    key: str,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for record in records:
        for name, value in _measurement_numeric_mapping(record, key).items():
            result[name] = result.get(name, 0.0) + value
    return result


def _sum_integer_mappings(
    records: tuple[Mapping[str, object], ...],
    key: str,
) -> dict[str, int]:
    result: dict[str, int] = {}
    for record in records:
        for name, value in _measurement_integer_mapping(record, key).items():
            result[name] = result.get(name, 0) + value
    return result


def _measurement_number(record: Mapping[str, object], key: str) -> float:
    value = record.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool) or value < 0:
        raise ValueError(f"Acceptance phase metric {key}が不正です")
    return float(value)


def _measurement_integer(record: Mapping[str, object], key: str) -> int:
    value = record.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"Acceptance phase metric {key}が不正です")
    return value


def _measurement_numeric_mapping(
    record: Mapping[str, object],
    key: str,
) -> dict[str, float]:
    value = record.get(key)
    if not isinstance(value, dict) or not all(
        isinstance(name, str)
        and isinstance(item, int | float)
        and not isinstance(item, bool)
        and item >= 0
        for name, item in value.items()
    ):
        raise ValueError(f"Acceptance phase metric {key}が不正です")
    return {name: float(item) for name, item in value.items()}


def _measurement_integer_mapping(
    record: Mapping[str, object],
    key: str,
) -> dict[str, int]:
    value = record.get(key)
    if not isinstance(value, dict) or not all(
        isinstance(name, str)
        and isinstance(item, int)
        and not isinstance(item, bool)
        and item >= 0
        for name, item in value.items()
    ):
        raise ValueError(f"Acceptance phase metric {key}が不正です")
    return cast(dict[str, int], value)


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
