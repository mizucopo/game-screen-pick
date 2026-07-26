"""固定3 workerとautoのtarget実測をprivacy-safeに比較する。"""

from collections.abc import Mapping
from typing import cast

from .acceptance_resource_budget import acceptance_run_resource_budget_passed


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


def _mapping(value: object, location: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"Video Scan comparison {location}がobjectではありません")
    return cast(dict[str, object], value)


def _positive_number(value: Mapping[str, object], key: str) -> float:
    result = value.get(key)
    if not isinstance(result, int | float) or isinstance(result, bool) or result <= 0:
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
