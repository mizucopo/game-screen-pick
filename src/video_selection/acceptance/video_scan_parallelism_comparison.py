"""固定3 workerとautoのtarget実測をprivacy-safeに比較する。"""

import math
from collections.abc import Mapping
from typing import cast

from .acceptance_resource_budget import acceptance_run_resource_budget_passed

_COMPARISON_IDENTITY_KEYS = (
    "configuration_digest",
    "effective_configuration_digest",
    "ollama_endpoint_identity",
    "model_identity_digest",
)


def build_video_scan_parallelism_comparison(
    fixed_three: Mapping[str, object],
    automatic: Mapping[str, object],
) -> dict[str, object]:
    """Stage同一性、resource、worker利用、Video Scan wall改善を判定する。"""
    fixed_diagnostics = _mapping(
        fixed_three.get("video_scan_parallelism"),
        "fixed3 video_scan_parallelism",
    )
    auto_diagnostics = _mapping(
        automatic.get("video_scan_parallelism"),
        "auto video_scan_parallelism",
    )
    fixed_wall = _positive_number(
        fixed_diagnostics,
        "scan_wall_seconds",
    )
    auto_wall = _positive_number(
        auto_diagnostics,
        "scan_wall_seconds",
    )
    fixed_artifact_digest = _digest(
        fixed_three,
        "stage_artifact_content_digest",
    )
    auto_artifact_digest = _digest(
        automatic,
        "stage_artifact_content_digest",
    )
    gates = {
        "execution_context_equal": video_scan_runs_share_comparison_context(
            fixed_three,
            automatic,
        ),
        "fixed_three_workers": (
            fixed_diagnostics.get("mode") == "fixed"
            and fixed_diagnostics.get("configured_workers") == 3
            and fixed_diagnostics.get("initial_workers") == 3
            and fixed_diagnostics.get("peak_workers") == 3
        ),
        "auto_exceeded_three_workers": (
            auto_diagnostics.get("mode") == "auto"
            and auto_diagnostics.get("configured_workers") == "auto"
            and _integer(auto_diagnostics, "peak_workers") > 3
        ),
        "stage_artifacts_equal": fixed_artifact_digest == auto_artifact_digest,
        "resource_budget": (
            _measurement_complete(fixed_diagnostics)
            and _measurement_complete(auto_diagnostics)
            and acceptance_run_resource_budget_passed(fixed_three)
            and acceptance_run_resource_budget_passed(automatic)
        ),
        "wall_time_improved": auto_wall < fixed_wall,
    }
    improvement_seconds = max(0.0, fixed_wall - auto_wall)
    return {
        "fixed3_video_scan_wall_seconds": fixed_wall,
        "auto_video_scan_wall_seconds": auto_wall,
        "wall_time_improvement_seconds": improvement_seconds,
        "wall_time_improvement_ratio": improvement_seconds / fixed_wall,
        "auto_initial_workers": _integer(auto_diagnostics, "initial_workers"),
        "auto_peak_workers": _integer(auto_diagnostics, "peak_workers"),
        "gates": gates,
        "passed": all(gates.values()),
    }


def video_scan_run_matches_comparison_context(
    run_record: Mapping[str, object],
    execution_context: Mapping[str, object],
) -> bool:
    """runの全attemptが現在のVideo Scan Comparison Contextと一致するか返す。"""
    return acceptance_run_matches_evidence_context(
        run_record,
        execution_context,
    )


def acceptance_run_matches_evidence_context(
    run_record: Mapping[str, object],
    execution_context: Mapping[str, object],
) -> bool:
    """runの全attemptが現在のAcceptance Evidence Contextと一致するか返す。"""
    try:
        expected = _video_scan_comparison_context(execution_context)
        contexts = _run_execution_contexts(run_record)
        return all(
            _video_scan_comparison_context(context) == expected for context in contexts
        )
    except ValueError:
        return False


def video_scan_runs_share_comparison_context(
    *run_records: Mapping[str, object],
) -> bool:
    """全runの全attemptが同じVideo Scan Comparison Contextか返す。"""
    try:
        contexts = tuple(
            context
            for run_record in run_records
            for context in _run_execution_contexts(run_record)
        )
        if not contexts:
            return False
        expected = _video_scan_comparison_context(contexts[0])
        return all(
            _video_scan_comparison_context(context) == expected
            for context in contexts[1:]
        )
    except ValueError:
        return False


def _run_execution_contexts(
    run_record: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    attempts = run_record.get("attempts")
    if isinstance(attempts, list) and attempts:
        return tuple(
            _mapping(
                _mapping(attempt, "run attempt").get("execution_context"),
                "run attempt execution_context",
            )
            for attempt in attempts
        )
    return (
        _mapping(
            run_record.get("execution_context"),
            "run execution_context",
        ),
    )


def _video_scan_comparison_context(
    execution_context: Mapping[str, object],
) -> dict[str, object]:
    identity = _mapping(
        execution_context.get("identity"),
        "execution context identity",
    )
    target = _mapping(
        execution_context.get("target"),
        "execution context target",
    )
    return {
        "identity": {key: identity.get(key) for key in _COMPARISON_IDENTITY_KEYS},
        "target": {
            key: value for key, value in target.items() if key != "visible_ram_bytes"
        },
    }


def _mapping(value: object, location: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"Video Scan comparison {location}がobjectではありません")
    return cast(dict[str, object], value)


def _positive_number(value: Mapping[str, object], key: str) -> float:
    result = value.get(key)
    if (
        not isinstance(result, int | float)
        or isinstance(result, bool)
        or not math.isfinite(result)
        or result <= 0
    ):
        raise ValueError(f"Video Scan comparison {key}が正のnumberではありません")
    return float(result)


def _integer(value: Mapping[str, object], key: str) -> int:
    result = value.get(key)
    if not isinstance(result, int) or isinstance(result, bool) or result < 1:
        raise ValueError(f"Video Scan comparison {key}が正のintegerではありません")
    return result


def _digest(value: Mapping[str, object], key: str) -> str:
    result = value.get(key)
    if (
        not isinstance(result, str)
        or len(result) != 64
        or any(character not in "0123456789abcdef" for character in result)
    ):
        raise ValueError(f"Video Scan comparison {key}がdigestではありません")
    return result


def _measurement_complete(value: Mapping[str, object]) -> bool:
    result = value.get("measurement_complete")
    return result is None or result is True
