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


def test_partially_offloaded_ollama_model_fails_automatic_gate() -> None:
    """Ollama modelが100% GPU residentでなければacceptanceが失敗すること。

    Arrange:
        - cold phaseでOllama modelの一部CPU offloadが観測される
    Act:
        - acceptance recordが構築される
    Assert:
        - fully resident gateがfalseでstatusがfailedになること
    """
    # Arrange / Act
    record = _build_record(
        human_quality={"status": "passed", "gates": {}},
        cold_overrides={"ollama_model_fully_resident": False},
    )

    # Assert
    gates = record["automatic_gates"]
    assert isinstance(gates, dict)
    assert gates["ollama_model_fully_resident"] is False
    assert record["status"] == "failed"


@pytest.mark.parametrize("duration_seconds", [float("nan"), float("inf"), -1.0])
def test_invalid_phase_duration_is_rejected(duration_seconds: float) -> None:
    """非有限または負のphase時間が合格判定へ使われないこと。

    Arrange:
        - 不正なcold phase durationが用意される
    Act:
        - acceptance recordが構築される
    Assert:
        - 非負の有限numberではないmetricとして拒否されること
    """
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="非負の有限number"):
        _build_record(
            human_quality={"status": "passed", "gates": {}},
            cold_overrides={"duration_seconds": duration_seconds},
        )


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


def test_normalized_baseline_removes_nested_attempt_source_revisions(
    tmp_path: Path,
) -> None:
    """attempt内のsource revision差がbaselineから除外されること。

    Arrange:
        - cold/warm attemptのcommitだけが異なる2つの合格recordが用意される
    Act:
        - 各recordからnormalized baselineが生成される
    Assert:
        - nested source revisionを含まず両baselineのbytesが一致すること
    """
    # Arrange
    record = _build_record(
        human_quality={"status": "passed", "gates": {"quality": True}},
    )
    phases = record["phases"]
    assert isinstance(phases, dict)
    for phase_name in ("cold", "warm"):
        phase = phases[phase_name]
        assert isinstance(phase, dict)
        phase["attempts"] = [
            {
                "execution_context": {
                    "source_revision": {"commit": "1" * 40, "dirty": False},
                    "identity": {
                        "commit": "1" * 40,
                        "configuration_digest": "a" * 64,
                    },
                }
            }
        ]
    first_directory = tmp_path / "first"
    second_directory = tmp_path / "second"

    # Act
    first_json, _first_markdown = write_normalized_baseline(
        record,
        first_directory,
    )
    for phase_name in ("cold", "warm"):
        phase = phases[phase_name]
        assert isinstance(phase, dict)
        attempts = phase["attempts"]
        assert isinstance(attempts, list)
        attempt = attempts[0]
        assert isinstance(attempt, dict)
        context = attempt["execution_context"]
        assert isinstance(context, dict)
        source_revision = context["source_revision"]
        identity = context["identity"]
        assert isinstance(source_revision, dict)
        assert isinstance(identity, dict)
        source_revision["commit"] = "2" * 40
        identity["commit"] = "2" * 40
    second_json, _second_markdown = write_normalized_baseline(
        record,
        second_directory,
    )

    # Assert
    first_bytes = first_json.read_bytes()
    assert first_bytes == second_json.read_bytes()
    assert b"source_revision" not in first_bytes
    assert b'"commit"' not in first_bytes


def test_unused_speech_runtime_is_explicit_and_consistent() -> None:
    """cold/warmでSTT未実行の場合もacceptance recordが生成されること。

    Arrange:
        - Speech Runtime Identityが両phaseでnullのSTT未実行結果が用意される
    Act:
        - acceptance recordが構築される
    Assert:
        - STT未使用がnullで公開され、runtime consistency gateが合格すること
    """
    # Arrange
    unused_runtime = {"speech_runtime_identity": None}

    # Act
    record = _build_record(
        human_quality={"status": "pending_human_review", "gates": {}},
        cold_overrides=unused_runtime,
        warm_overrides=unused_runtime,
    )

    # Assert
    assert record["runtime"] == {"speech_to_text": None}
    consistency = record["consistency"]
    assert isinstance(consistency, dict)
    assert consistency["speech_runtime_identity_equal"] is True
    gates = record["automatic_gates"]
    assert isinstance(gates, dict)
    assert gates["speech_runtime_identity_consistency"] is True


def _build_record(
    *,
    human_quality: Mapping[str, object],
    cold_overrides: Mapping[str, object] | None = None,
    warm_overrides: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """release acceptance recordの共通入力からrecordを返す。"""
    phase: dict[str, object] = {
        "duration_seconds": 10.0,
        "unexpected_recompute_count": 0,
        "persistent_cache_bytes": 1024,
        "peak_additional_bytes": 2048,
        "ollama_global_gpu_peak_mib": 1000,
        "stt_non_ollama_gpu_peak_mib": 1000,
        "ollama_model_observed": True,
        "ollama_model_fully_resident": True,
        "resource_sampling_complete": True,
        "speech_runtime_identity": "speech_" + "d" * 64,
        "normalized_result_digest": "a" * 64,
    }
    cold = dict(phase)
    if cold_overrides is not None:
        cold.update(cold_overrides)
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
        storage_preflight={
            "input_video_bytes": 1024,
            "input_video_count": 3,
            "artifact_available_bytes": 200 * 1024**3,
            "required_artifact_capacity_bytes": 160 * 1024**3,
        },
        video_set={"fingerprint": "c" * 64, "scenario_count": 3},
        cold=cold,
        warm=warm,
        human_quality=human_quality,
    )
