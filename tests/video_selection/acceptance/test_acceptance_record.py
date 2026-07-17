"""versioned acceptance recordとprivacy gateのtest。"""

from collections.abc import Mapping
from pathlib import Path

import pytest

from src.video_selection.acceptance.acceptance_record import (
    build_acceptance_record,
    validate_acceptance_record_privacy,
    write_normalized_baseline,
)


def test_phase_budgets_and_human_quality_determine_pending_pass_failure() -> None:
    """phase budgetとhuman statusからacceptance statusが決まること。

    Arrange:
        - release予算内のcold/warmとpending human aggregateが用意される
    Act:
        - acceptance recordが構築される
    Assert:
        - automatic gateは全合格しstatusだけがpendingになること
        - human pass後は全体statusがpassedになること
    """
    # Arrange
    pending_quality: dict[str, object] = {
        "status": "pending_human_review",
        "gates": {},
    }
    passed_quality: dict[str, object] = {
        "status": "passed",
        "gates": {"quality": True},
    }

    # Act
    pending = _build_record(human_quality=pending_quality)
    passed = _build_record(human_quality=passed_quality)

    # Assert
    assert pending["status"] == "pending_human_review"
    automatic_gates = pending["automatic_gates"]
    assert isinstance(automatic_gates, dict)
    assert all(automatic_gates.values())
    assert passed["status"] == "passed"


def test_warm_recompute_or_budget_excess_fails_automatic_gate() -> None:
    """warm recomputeまたは性能超過がtimeoutでなくgate failureになること。

    Arrange:
        - release warmで1件再計算し181秒かかったphaseが用意される
    Act:
        - acceptance recordが構築される
    Assert:
        - warm duration/recompute gateがfalseでstatusがfailedになること
    """
    # Arrange
    warm_overrides: dict[str, object] = {
        "duration_seconds": 181.0,
        "unexpected_recompute_count": 1,
    }

    # Act
    record = _build_record(
        human_quality={"status": "passed", "gates": {}},
        warm_overrides=warm_overrides,
    )

    # Assert
    assert record["status"] == "failed"
    automatic_gates = record["automatic_gates"]
    assert isinstance(automatic_gates, dict)
    assert automatic_gates["warm_duration"] is False
    assert automatic_gates["warm_unexpected_recompute"] is False


def test_actual_paths_and_video_names_are_rejected_from_record() -> None:
    """actual pathまたはprivate video名がacceptance recordへ混入すると拒否されること。

    Arrange:
        - target fieldへ実input pathを混入したrecordが用意される
    Act:
        - privacy gateが実行される
    Assert:
        - private valueとしてValueErrorになること
    """
    # Arrange
    record = _build_record(
        human_quality={"status": "pending_human_review", "gates": {}},
    )
    target = record["target"]
    assert isinstance(target, dict)
    target["location"] = "/mnt/g/private/movie"

    # Act / Assert
    with pytest.raises(ValueError, match="private value"):
        validate_acceptance_record_privacy(
            record,
            forbidden_values=("/mnt/g/private/movie", "secret-video.mkv"),
        )


def test_passed_record_generates_normalized_json_and_markdown(tmp_path: Path) -> None:
    """合格済みrecordからcommit候補baselineが2形式で生成されること。

    Arrange:
        - automatic/human gate合格済みrecordが用意される
    Act:
        - normalized baselineが生成される
    Assert:
        - JSONとMarkdownが存在しsource commitがbaselineから除かれること
    """
    # Arrange
    record = _build_record(
        human_quality={"status": "passed", "gates": {"quality": True}},
    )

    # Act
    json_path, markdown_path = write_normalized_baseline(record, tmp_path)

    # Assert
    assert json_path.is_file()
    assert markdown_path.is_file()
    assert "source_revision" not in json_path.read_text(encoding="utf-8")
    assert "Normalized digest" in markdown_path.read_text(encoding="utf-8")


def _build_record(
    *,
    human_quality: Mapping[str, object],
    warm_overrides: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """release acceptance recordの共通入力からrecordを返す。"""
    phase: dict[str, object] = {
        "duration_seconds": 10.0,
        "unexpected_recompute_count": 0,
        "persistent_cache_bytes": 1024,
        "peak_additional_bytes": 2048,
        "ollama_global_gpu_peak_mib": 1000,
        "stt_global_gpu_peak_mib": 1000,
        "resource_sampling_complete": True,
        "normalized_result_digest": "a" * 64,
    }
    warm = dict(phase)
    if warm_overrides is not None:
        warm.update(warm_overrides)
    return build_acceptance_record(
        suite="release",
        commit="b" * 40,
        dirty=False,
        target={"os": "windows_11_wsl2", "gpu": "rtx_5090"},
        configuration={"image_count": 100, "spoiler_sensitivity": "medium"},
        models={"scene_catalog": {"execution_identity": "sha256:abc"}},
        video_set={"fingerprint": "c" * 64, "scenario_count": 3},
        cold=dict(phase),
        warm=warm,
        human_quality=human_quality,
    )
