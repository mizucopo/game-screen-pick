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
        "execution_context_equal": True,
        "fixed_three_workers": True,
        "auto_exceeded_three_workers": True,
        "stage_artifacts_equal": True,
        "resource_budget": True,
        "wall_time_improved": True,
    }
    assert comparison["passed"] is True
    assert comparison["wall_time_improvement_seconds"] == 40.0
    assert comparison["wall_time_improvement_ratio"] == pytest.approx(1 / 3)


def test_different_execution_contexts_fail_the_comparison() -> None:
    """異なるtarget execution contextのwall timeが比較されないこと。

    Arrange:
        - CPU identityだけが異なるfixed3とauto runが用意される
    Act:
        - Video Scan parallelism比較が構築される
    Assert:
        - execution context gateと比較全体が不合格になること
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
    automatic["execution_context"] = _execution_context(cpu="changed")

    # Act
    comparison = build_video_scan_parallelism_comparison(fixed, automatic)

    # Assert
    gates = comparison["gates"]
    assert isinstance(gates, dict)
    assert gates["execution_context_equal"] is False
    assert comparison["passed"] is False


def test_commit_change_preserves_the_comparison_context() -> None:
    """commitだけが異なる再開attemptが同じ比較条件として扱われること。

    Arrange:
        - 設定、model、targetが同じでsource commitだけが異なるrunが用意される
    Act:
        - Video Scan parallelism比較が構築される
    Assert:
        - execution context gateと比較全体が合格されること
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
    automatic["execution_context"] = _execution_context(commit="e" * 40)

    # Act
    comparison = build_video_scan_parallelism_comparison(fixed, automatic)

    # Assert
    gates = comparison["gates"]
    assert isinstance(gates, dict)
    assert gates["execution_context_equal"] is True
    assert comparison["passed"] is True


@pytest.mark.parametrize(
    "identity_key",
    (
        "configuration_digest",
        "effective_configuration_digest",
        "ollama_endpoint_identity",
        "model_identity_digest",
    ),
)
def test_comparison_identity_change_breaks_the_comparison(
    identity_key: str,
) -> None:
    """比較対象identityが異なるrunが比較不能にされること。

    Arrange:
        - targetとcommitが同じで比較対象identityだけが異なるrunが用意される
    Act:
        - Video Scan parallelism比較が構築される
    Assert:
        - execution context gateと比較全体が不合格にされること
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
    changed_context = _execution_context()
    changed_identity = changed_context["identity"]
    assert isinstance(changed_identity, dict)
    changed_identity[identity_key] = "e" * 64
    automatic["execution_context"] = changed_context

    # Act
    comparison = build_video_scan_parallelism_comparison(fixed, automatic)

    # Assert
    gates = comparison["gates"]
    assert isinstance(gates, dict)
    assert gates["execution_context_equal"] is False
    assert comparison["passed"] is False


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
            {"ollama_model_observed": False},
            "resource_budget",
            id="ollama-model-not-observed",
        ),
        pytest.param(
            {"ollama_model_fully_resident": False},
            "resource_budget",
            id="ollama-model-partially-offloaded",
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


@pytest.mark.parametrize("scan_wall_seconds", [float("nan"), float("inf")])
def test_nonfinite_scan_wall_time_is_rejected(scan_wall_seconds: float) -> None:
    """非有限Video Scan時間が並列改善として受理されないこと。

    Arrange:
        - 非有限wall timeを持つauto runが用意される
    Act:
        - fixed3とautoの並列比較が構築される
    Assert:
        - 正の有限numberではない値として拒否されること
    """
    # Arrange
    fixed = _run_record(
        workers=3,
        mode="fixed",
        wall_seconds=100.0,
        artifact_digest="a" * 64,
    )
    automatic = _run_record(
        workers=6,
        mode="auto",
        wall_seconds=scan_wall_seconds,
        artifact_digest="a" * 64,
    )

    # Act
    # Assert
    with pytest.raises(ValueError, match="正のnumber"):
        build_video_scan_parallelism_comparison(fixed, automatic)


def _run_record(
    *,
    workers: int,
    mode: str,
    wall_seconds: float,
    artifact_digest: str,
) -> dict[str, object]:
    """比較test用のprivacy-safe Acceptance Run recordを返す。"""
    return {
        "execution_context": _execution_context(),
        "stage_artifact_content_digest": artifact_digest,
        "resource_sampling_complete": True,
        "ollama_model_observed": True,
        "ollama_model_fully_resident": True,
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


def _execution_context(
    *,
    cpu: str = "stable",
    commit: str = "d" * 40,
) -> dict[str, object]:
    """比較test用のprivacy-safe execution contextを返す。"""
    return {
        "identity": {
            "configuration_digest": "f" * 64,
            "effective_configuration_digest": "a" * 64,
            "ollama_endpoint_identity": "b" * 64,
            "model_identity_digest": "c" * 64,
            "commit": commit,
        },
        "source_revision": {"commit": commit, "dirty": False},
        "target": {
            "cpu": cpu,
            "logical_cpu_count": 24,
            "visible_ram_bytes": 64 * 1024**3,
        },
        "configuration": {},
        "models": {},
    }
