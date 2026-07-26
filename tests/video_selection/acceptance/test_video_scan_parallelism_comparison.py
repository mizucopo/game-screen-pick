"""Video Scan parallelism target比較gateのtest。"""

import pytest

from src.video_selection.acceptance.video_scan_parallelism_comparison import (
    build_video_scan_parallelism_comparison,
)


def test_fixed_three_and_faster_auto_pass_all_comparison_gates() -> None:
    """同一成果物を高速に生成するauto runが比較gateを通過すること。

    Arrange:
        - resource予算内の固定3 runと6 workerを利用したauto runが用意される
    Act:
        - Video Scan parallelism比較が構築される
    Assert:
        - 成果物、resource、worker利用、wall改善の全gateが合格すること
    """
    # Arrange
    fixed = _run_record(
        workers=3,
        mode="fixed",
        wall_seconds=120.0,
        artifact_digest="a" * 64,
    )
    automatic = _run_record(
        workers=6,
        mode="auto",
        wall_seconds=80.0,
        artifact_digest="a" * 64,
    )

    # Act
    comparison = build_video_scan_parallelism_comparison(fixed, automatic)

    # Assert
    assert comparison["gates"] == {
        "fixed_three_workers": True,
        "auto_exceeded_three_workers": True,
        "stage_artifacts_equal": True,
        "resource_budget": True,
        "wall_time_improved": True,
    }
    assert comparison["passed"] is True
    assert comparison["wall_time_improvement_seconds"] == 40.0
    assert comparison["wall_time_improvement_ratio"] == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    ("automatic_change", "failed_gate"),
    [
        pytest.param(
            {
                "video_scan_parallelism": {
                    "mode": "auto",
                    "configured_workers": "auto",
                    "initial_workers": 3,
                    "peak_workers": 6,
                    "scan_wall_seconds": 120.0,
                }
            },
            "wall_time_improved",
            id="same-wall-time",
        ),
        pytest.param(
            {"stage_artifact_content_digest": "b" * 64},
            "stage_artifacts_equal",
            id="different-artifacts",
        ),
        pytest.param(
            {"resource_sampling_complete": False},
            "resource_budget",
            id="incomplete-resource-sampling",
        ),
        pytest.param(
            {
                "video_scan_parallelism": {
                    "mode": "auto",
                    "configured_workers": "auto",
                    "initial_workers": 3,
                    "peak_workers": 6,
                    "scan_wall_seconds": 80.0,
                    "measurement_complete": False,
                }
            },
            "resource_budget",
            id="incomplete-parallelism-measurement",
        ),
        pytest.param(
            {
                "video_scan_parallelism": {
                    "mode": "auto",
                    "configured_workers": "auto",
                    "initial_workers": 3,
                    "peak_workers": 3,
                    "scan_wall_seconds": 80.0,
                }
            },
            "auto_exceeded_three_workers",
            id="auto-never-exceeds-three",
        ),
    ],
)
def test_failed_comparison_dimension_is_reported(
    automatic_change: dict[str, object],
    failed_gate: str,
) -> None:
    """比較条件を満たさないrunが該当gateで不合格にされること。

    Arrange:
        - 合格する固定3 runと一つの条件だけを満たさないauto runが用意される
    Act:
        - Video Scan parallelism比較が構築される
    Assert:
        - 指定したgateと比較全体だけが不合格として記録されること
    """
    # Arrange
    fixed = _run_record(
        workers=3,
        mode="fixed",
        wall_seconds=120.0,
        artifact_digest="a" * 64,
    )
    automatic = {
        **_run_record(
            workers=6,
            mode="auto",
            wall_seconds=80.0,
            artifact_digest="a" * 64,
        ),
        **automatic_change,
    }

    # Act
    comparison = build_video_scan_parallelism_comparison(fixed, automatic)

    # Assert
    gates = comparison["gates"]
    assert isinstance(gates, dict)
    assert gates[failed_gate] is False
    assert comparison["passed"] is False


def _run_record(
    *,
    workers: int,
    mode: str,
    wall_seconds: float,
    artifact_digest: str,
) -> dict[str, object]:
    """比較test用のprivacy-safe Acceptance Run recordを返す。"""
    return {
        "stage_artifact_content_digest": artifact_digest,
        "resource_sampling_complete": True,
        "persistent_cache_bytes": 1024,
        "peak_additional_bytes": 2048,
        "ollama_global_gpu_peak_mib": 1000,
        "stt_non_ollama_gpu_peak_mib": 1000,
        "video_scan_parallelism": {
            "mode": mode,
            "configured_workers": workers if mode == "fixed" else "auto",
            "initial_workers": 3,
            "peak_workers": workers,
            "scan_wall_seconds": wall_seconds,
        },
    }
