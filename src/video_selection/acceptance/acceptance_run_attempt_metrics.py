"""中断をまたぐAcceptance Run Attemptの計測検証と集計。"""

import math
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
    "ollama_sample_count",
    "ollama_sample_error_count",
)
_LEGACY_OPTIONAL_INTEGER_METRICS = frozenset(
    {
        "ollama_sample_count",
        "ollama_sample_error_count",
    }
)
_MAXIMUM_INTEGER_METRICS = (
    "persistent_cache_bytes",
    "peak_additional_bytes",
    "system_global_gpu_peak_mib",
    "ollama_global_gpu_peak_mib",
    "stt_non_ollama_gpu_peak_mib",
    "ollama_model_size_bytes",
    "ollama_model_size_vram_bytes",
)
_BASELINE_INTEGER_METRICS = (
    "process_gpu_baseline_mib",
    "system_gpu_baseline_mib",
)


def build_incomplete_interrupt_attempt(
    duration_seconds: float,
    recovered_metrics: Mapping[str, object] | None = None,
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
    if recovered_metrics is not None:
        for key in (
            "cache_hit_count",
            "cache_miss_count",
            "reuse_count",
            "unexpected_recompute_count",
            "stage_durations_seconds",
            "completed_stage_counts",
        ):
            if key in recovered_metrics:
                record[key] = recovered_metrics[key]
    validate_run_measurements(record)
    return record


def validate_run_measurements(
    record: Mapping[str, object],
    *,
    allow_legacy_ollama_counts: bool = False,
) -> None:
    """一試行を安全に累積できる完全な計測recordとして検証する。"""
    _measurement_number(record, "duration_seconds")
    for key in (
        *_SUMMED_INTEGER_METRICS,
        *_MAXIMUM_INTEGER_METRICS,
        *_BASELINE_INTEGER_METRICS,
    ):
        if allow_legacy_ollama_counts:
            _measurement_integer_compatible_with_legacy(record, key)
        else:
            _measurement_integer(record, key)
    _measurement_numeric_mapping(record, "stage_durations_seconds")
    _measurement_integer_mapping(record, "completed_stage_counts")
    for key in (
        "resource_sampling_complete",
        "ollama_model_observed",
        "ollama_model_fully_resident",
    ):
        if not isinstance(record.get(key), bool):
            raise ValueError(f"Acceptance run metric {key}が不正です")


def aggregate_run_attempts(
    records: tuple[Mapping[str, object], ...],
) -> dict[str, object]:
    """全試行の作業量と保守的resource値を成功recordへ集約する。"""
    if not records:
        raise ValueError("Acceptance run attemptがありません")
    for record in records:
        validate_run_measurements(record, allow_legacy_ollama_counts=True)
    aggregate = dict(records[-1])
    aggregate["attempt_count"] = len(records)
    aggregate["attempts"] = [dict(record) for record in records]
    aggregate["duration_seconds"] = sum(
        _measurement_number(record, "duration_seconds") for record in records
    )
    for key in _SUMMED_INTEGER_METRICS:
        aggregate[key] = sum(
            _measurement_integer_compatible_with_legacy(record, key)
            for record in records
        )
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
    parallelism = _aggregate_video_scan_parallelism(records)
    if parallelism is not None:
        aggregate["video_scan_parallelism"] = parallelism
    aggregate.pop("failure_reason", None)
    aggregate.pop("failure_exit_code", None)
    return aggregate


def _aggregate_video_scan_parallelism(
    records: tuple[Mapping[str, object], ...],
) -> dict[str, object] | None:
    diagnostics = tuple(
        value
        for record in records
        if isinstance(
            value := record.get("video_scan_parallelism"),
            dict,
        )
    )
    if not diagnostics:
        return None
    if len(records) == 1:
        return dict(diagnostics[0])
    result = dict(diagnostics[-1])
    contexts_are_equal = True
    for key in (
        "mode",
        "configured_workers",
        "decode_backend",
        "auto_max_workers",
    ):
        values = {str(item.get(key)) for item in diagnostics}
        if len(values) != 1:
            contexts_are_equal = False
    result["initial_workers"] = _parallelism_integer(
        diagnostics[0],
        "initial_workers",
    )
    result["final_workers"] = _parallelism_integer(
        diagnostics[-1],
        "final_workers",
    )
    result["peak_workers"] = max(
        _parallelism_integer(item, "peak_workers") for item in diagnostics
    )
    result["completed_scans"] = sum(
        _parallelism_nonnegative_integer(item, "completed_scans")
        for item in diagnostics
    )
    result["scan_wall_seconds"] = round(
        sum(_parallelism_number(item, "scan_wall_seconds") for item in diagnostics),
        3,
    )
    result["attempt_count"] = len(records)
    result["measurement_complete"] = (
        len(diagnostics) == len(records) and contexts_are_equal
    )
    changes: list[dict[str, object]] = []
    elapsed_offset = 0.0
    for attempt_index, item in enumerate(diagnostics, start=1):
        values = item.get("changes", [])
        if not isinstance(values, list) or any(
            not isinstance(value, dict) for value in values
        ):
            raise ValueError("Acceptance run Video Scan changesが不正です")
        for value in values:
            change = dict(value)
            elapsed = change.get("elapsed_seconds")
            if isinstance(elapsed, int | float) and not isinstance(elapsed, bool):
                change["elapsed_seconds"] = round(elapsed_offset + float(elapsed), 3)
            change["attempt"] = attempt_index
            changes.append(change)
        elapsed_offset += _parallelism_number(item, "scan_wall_seconds")
    result["changes"] = changes
    return result


def _parallelism_integer(value: Mapping[str, object], key: str) -> int:
    result = value.get(key)
    if not isinstance(result, int) or isinstance(result, bool) or result < 1:
        raise ValueError(f"Acceptance run Video Scan {key}が不正です")
    return result


def _parallelism_nonnegative_integer(
    value: Mapping[str, object],
    key: str,
) -> int:
    result = value.get(key)
    if not isinstance(result, int) or isinstance(result, bool) or result < 0:
        raise ValueError(f"Acceptance run Video Scan {key}が不正です")
    return result


def _parallelism_number(value: Mapping[str, object], key: str) -> float:
    result = value.get(key)
    if (
        not isinstance(result, int | float)
        or isinstance(result, bool)
        or not math.isfinite(result)
        or result < 0
    ):
        raise ValueError(f"Acceptance run Video Scan {key}が不正です")
    return float(result)


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
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"Acceptance run metric {key}が不正です")
    return float(value)


def _measurement_integer(record: Mapping[str, object], key: str) -> int:
    value = record.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"Acceptance run metric {key}が不正です")
    return value


def _measurement_integer_compatible_with_legacy(
    record: Mapping[str, object],
    key: str,
) -> int:
    """旧attemptに存在しない追加metricだけを0として扱う。"""
    if key in _LEGACY_OPTIONAL_INTEGER_METRICS and key not in record:
        return 0
    return _measurement_integer(record, key)


def _measurement_numeric_mapping(
    record: Mapping[str, object],
    key: str,
) -> dict[str, float]:
    value = record.get(key)
    if not isinstance(value, dict) or not all(
        isinstance(name, str)
        and isinstance(item, int | float)
        and not isinstance(item, bool)
        and math.isfinite(item)
        and item >= 0
        for name, item in value.items()
    ):
        raise ValueError(f"Acceptance run metric {key}が不正です")
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
        raise ValueError(f"Acceptance run metric {key}が不正です")
    return cast(dict[str, int], value)
