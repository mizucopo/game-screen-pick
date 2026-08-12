"""Acceptance Run Attempt集計のtest。"""

import pytest

from src.video_selection.acceptance.acceptance_run_attempt_metrics import (
    aggregate_run_attempts,
    validate_run_measurements,
)


def test_parallelism_change_preserves_attempts_as_incomplete_measurement() -> None:
    """再開時の並列設定変更が試行を失わず比較不能として集約されること。

    Arrange:
        - auto上限6と8で実行された二つの完了attemptが用意される
    Act:
        - 両attemptが一つのrun recordへ集約される
    Assert:
        - 設定不一致で例外にせずmeasurement incompleteになること
        - 各attemptの設定と実行contextがそのまま保持されること
    """
    # Arrange
    first = _attempt(auto_max_workers=6, context_name="before-restart")
    second = _attempt(auto_max_workers=8, context_name="after-restart")

    # Act
    aggregate = aggregate_run_attempts((first, second))

    # Assert
    parallelism = aggregate["video_scan_parallelism"]
    assert isinstance(parallelism, dict)
    assert parallelism["measurement_complete"] is False
    assert parallelism["auto_max_workers"] == 8
    attempts = aggregate["attempts"]
    assert isinstance(attempts, list)
    assert [item["execution_context"] for item in attempts] == [
        {"name": "before-restart"},
        {"name": "after-restart"},
    ]


def test_independent_gpu_channel_counts_are_summed_across_attempts() -> None:
    """system GPUとOllamaの観測件数が独立して合算されること。

    Arrange:
        - system GPUとOllamaで異なる観測件数を持つ二つのattemptが用意される
    Act:
        - 両attemptが一つのrun recordへ集約される
    Assert:
        - 各観測channelの成功数と失敗数が混同されず合算されること
    """
    # Arrange
    first = _attempt(auto_max_workers=6, context_name="first")
    first.update(
        {
            "gpu_sample_count": 10,
            "gpu_sample_error_count": 1,
            "ollama_sample_count": 8,
            "ollama_sample_error_count": 3,
        }
    )
    second = _attempt(auto_max_workers=6, context_name="second")
    second.update(
        {
            "gpu_sample_count": 20,
            "gpu_sample_error_count": 2,
            "ollama_sample_count": 17,
            "ollama_sample_error_count": 5,
        }
    )

    # Act
    aggregate = aggregate_run_attempts((first, second))

    # Assert
    assert aggregate["gpu_sample_count"] == 30
    assert aggregate["gpu_sample_error_count"] == 3
    assert aggregate["ollama_sample_count"] == 25
    assert aggregate["ollama_sample_error_count"] == 8


def test_legacy_incomplete_attempt_is_not_promoted_by_new_channel_counts() -> None:
    """旧attemptの不完全なresource証拠が新field追加で格上げされないこと。

    Arrange:
        - Ollama観測件数を持たずresource計測が不完全な旧attemptが用意される
    Act:
        - 旧attemptと新attemptが一つのrun recordへ集約される
    Assert:
        - 旧件数は0として読まれrun全体はresource計測不完全のままになること
    """
    # Arrange
    legacy = _attempt(auto_max_workers=6, context_name="legacy")
    legacy.pop("ollama_sample_count")
    legacy.pop("ollama_sample_error_count")
    legacy["resource_sampling_complete"] = False
    current = _attempt(auto_max_workers=6, context_name="current")

    # Act
    aggregate = aggregate_run_attempts((legacy, current))

    # Assert
    assert aggregate["ollama_sample_count"] == 1
    assert aggregate["ollama_sample_error_count"] == 0
    assert aggregate["resource_sampling_complete"] is False


def test_current_attempt_requires_independent_ollama_channel_counts() -> None:
    """新しいattempt計測ではOllama channel件数が必須になること。

    Arrange:
        - Ollama観測件数だけを欠く新規attemptが用意される
    Act:
        - 一試行の計測recordが検証される
    Assert:
        - 新規計測の欠落として拒否されること
    """
    # Arrange
    attempt = _attempt(auto_max_workers=6, context_name="current")
    attempt.pop("ollama_sample_count")

    # Act
    # Assert
    with pytest.raises(ValueError, match="ollama_sample_count"):
        validate_run_measurements(attempt)


@pytest.mark.parametrize("duration_seconds", (float("nan"), float("inf")))
def test_nonfinite_attempt_duration_is_rejected(duration_seconds: float) -> None:
    """非有限のattempt時間が集計前に拒否されること。

    Arrange:
        - NaNまたはInfinityのdurationを持つ完了attemptが用意される
    Act:
        - attemptがrun recordへ集約される
    Assert:
        - 再開回数をまたいだ不正な計測値として拒否されること
    """
    # Arrange
    attempt = _attempt(auto_max_workers=6, context_name="corrupt")
    attempt["duration_seconds"] = duration_seconds

    # Act
    # Assert
    with pytest.raises(ValueError, match="duration_seconds"):
        aggregate_run_attempts((attempt,))


def _attempt(
    *,
    auto_max_workers: int,
    context_name: str,
) -> dict[str, object]:
    """集計contractを満たす最小の完了attemptを返す。"""
    return {
        "operation_status": "completed",
        "duration_seconds": 1.0,
        "cache_hit_count": 0,
        "cache_miss_count": 1,
        "reuse_count": 0,
        "unexpected_recompute_count": 1,
        "disk_sample_count": 1,
        "disk_sample_error_count": 0,
        "gpu_sample_count": 1,
        "gpu_sample_error_count": 0,
        "ollama_sample_count": 1,
        "ollama_sample_error_count": 0,
        "persistent_cache_bytes": 1,
        "peak_additional_bytes": 1,
        "system_global_gpu_peak_mib": 1,
        "ollama_global_gpu_peak_mib": 1,
        "stt_non_ollama_gpu_peak_mib": 1,
        "ollama_model_size_bytes": 1,
        "ollama_model_size_vram_bytes": 1,
        "process_gpu_baseline_mib": 0,
        "system_gpu_baseline_mib": 0,
        "stage_durations_seconds": {"scan-video": 1.0},
        "completed_stage_counts": {"scan-video": 1},
        "resource_sampling_complete": True,
        "ollama_model_observed": True,
        "ollama_model_fully_resident": True,
        "execution_context": {"name": context_name},
        "video_scan_parallelism": {
            "mode": "auto",
            "configured_workers": "auto",
            "decode_backend": "nvdec",
            "auto_max_workers": auto_max_workers,
            "initial_workers": 3,
            "final_workers": auto_max_workers,
            "peak_workers": auto_max_workers,
            "completed_scans": 1,
            "scan_wall_seconds": 1.0,
            "changes": [],
        },
    }
