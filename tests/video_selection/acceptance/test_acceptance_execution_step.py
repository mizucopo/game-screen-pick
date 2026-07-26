"""Acceptance Execution Stepのtest。"""

from pathlib import Path

import pytest

from src.video_selection.acceptance.acceptance_execution_step import (
    AcceptanceExecutionStep,
)
from src.video_selection.models.effective_configuration import EffectiveConfiguration


def test_phase_and_comparison_use_separate_state_boundaries(
    tmp_path: Path,
) -> None:
    """phaseとcomparisonのstate境界が型付きstepから返されること。

    Arrange:
        - cold phaseとfixed3 comparisonの設定が用意される
    Act:
        - 両方のAcceptance Execution Stepが構築される
    Assert:
        - record、active、attempt、failureのstate keyが混在しないこと
    """
    # Arrange
    configuration = EffectiveConfiguration(
        video_input_folder=tmp_path / "input",
        output_folder=tmp_path / "output",
    )

    # Act
    phase = AcceptanceExecutionStep("phase", "cold", configuration)
    comparison = AcceptanceExecutionStep(
        "comparison",
        "fixed3",
        configuration,
    )

    # Assert
    assert phase.records_state_key == "phases"
    assert phase.active_state_key == "active_phase"
    assert phase.attempts_state_key == "phase_attempts"
    assert phase.failure_context == {"phase": "cold"}
    assert phase.is_cold_phase is True
    assert comparison.records_state_key == "comparison_runs"
    assert comparison.active_state_key == "active_comparison_run"
    assert comparison.attempts_state_key == "comparison_run_attempts"
    assert comparison.failure_context == {"comparison_run": "fixed3"}
    assert comparison.is_cold_phase is False


def test_invalid_name_for_execution_kind_is_rejected(tmp_path: Path) -> None:
    """Comparison RunがAcceptance Phase名として構築されないこと。

    Arrange:
        - target acceptance用の実効設定が用意される
    Act:
        - fixed3をphaseとして構築する操作が試行される
    Assert:
        - ubiquitous language違反としてValueErrorが返されること
    """
    # Arrange
    configuration = EffectiveConfiguration(
        video_input_folder=tmp_path / "input",
        output_folder=tmp_path / "output",
    )

    # Act
    with pytest.raises(ValueError) as error:
        AcceptanceExecutionStep("phase", "fixed3", configuration)

    # Assert
    assert "execution step" in str(error.value)
