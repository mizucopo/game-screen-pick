"""中断をまたぐAcceptance Phase Attemptの計測検証と集計。"""

from collections.abc import Mapping
from typing import cast

_SUMMED_INTEGER_METRICS = (
    "cache_hit_count",
    "cache_miss_count",
    "reuse_count",
    "unexpected_recompute_count",
    "disk_sample_count",
    "disk_sample_error_count",
    "gpu_sample_count",
    "gpu_sample_error_count",
)
_MAXIMUM_INTEGER_METRICS = (
    "persistent_cache_bytes",
    "peak_additional_bytes",
    "system_global_gpu_peak_mib",
    "ollama_global_gpu_peak_mib",
    "stt_global_gpu_peak_mib",
    "ollama_model_size_bytes",
    "ollama_model_size_vram_bytes",
)
_BASELINE_INTEGER_METRICS = (
    "process_gpu_baseline_mib",
    "system_gpu_baseline_mib",
)


def build_incomplete_interrupt_attempt(
    duration_seconds: float,
) -> dict[str, object]:
    """詳細計測を確定できないuser interruptを保守的な試行として返す。"""
    record: dict[str, object] = {
        "operation_status": "failed",
        "failure_reason": "user_interrupt",
        "failure_exit_code": 130,
        "duration_seconds": duration_seconds,
        "stage_durations_seconds": {},
        "completed_stage_counts": {},
        "resource_sampling_complete": False,
        "ollama_model_observed": False,
        "ollama_model_fully_resident": False,
        **dict.fromkeys(_SUMMED_INTEGER_METRICS, 0),
        **dict.fromkeys(_MAXIMUM_INTEGER_METRICS, 0),
        **dict.fromkeys(_BASELINE_INTEGER_METRICS, 0),
    }
    validate_phase_measurements(record)
    return record


def validate_phase_measurements(record: Mapping[str, object]) -> None:
    """一試行を安全に累積できる完全な計測recordとして検証する。"""
    _measurement_number(record, "duration_seconds")
    for key in (
        *_SUMMED_INTEGER_METRICS,
        *_MAXIMUM_INTEGER_METRICS,
        *_BASELINE_INTEGER_METRICS,
    ):
        _measurement_integer(record, key)
    _measurement_numeric_mapping(record, "stage_durations_seconds")
    _measurement_integer_mapping(record, "completed_stage_counts")
    for key in (
        "resource_sampling_complete",
        "ollama_model_observed",
        "ollama_model_fully_resident",
    ):
        if not isinstance(record.get(key), bool):
            raise ValueError(f"Acceptance phase metric {key}が不正です")


def aggregate_phase_attempts(
    records: tuple[Mapping[str, object], ...],
) -> dict[str, object]:
    """全試行の作業量と保守的resource値を成功recordへ集約する。"""
    if not records:
        raise ValueError("Acceptance phase attemptがありません")
    for record in records:
        validate_phase_measurements(record)
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
    aggregate["ollama_model_observed"] = any(
        record["ollama_model_observed"] is True for record in records
    )
    observed_records = tuple(
        record for record in records if record["ollama_model_observed"] is True
    )
    aggregate["ollama_model_fully_resident"] = bool(observed_records) and all(
        record["ollama_model_fully_resident"] is True for record in observed_records
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
