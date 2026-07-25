"""Acceptance phase実行境界のtest。"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.video_selection.acceptance.execute_acceptance_phase import (
    execute_acceptance_phase,
)
from src.video_selection.models.effective_configuration import EffectiveConfiguration
from src.video_selection.models.run_failure import RunFailure
from tests.video_selection.fakes.fake_model_runtime import FakeModelRuntime


def test_phase_duration_excludes_resource_monitor_shutdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """phase durationがresource monitor停止時間を含まず記録されること。

    Arrange:
        - pipeline完了時刻を20、monitor停止後時刻を100とするclockが用意される
    Act:
        - operation failureになるAcceptance phaseが実行される
    Assert:
        - 開始10からpipeline完了20までの10秒だけが記録されること
    """
    # Arrange
    now = [10.0]
    configuration = EffectiveConfiguration(
        video_input_folder=tmp_path / "input",
        output_folder=tmp_path / "output",
    )
    resolved_models = FakeModelRuntime("phase-timing").resolve_models(configuration)
    failure = RunFailure(
        reason_code="test_failure",
        exit_code=1,
        remediation_code="retry",
        resume_guidance="completed_stages_reusable",
    )

    def execute(_self: object, _operation: object) -> tuple[int, RunFailure]:
        now[0] = 20.0
        return 1, failure

    def stop_disk_monitor() -> dict[str, object]:
        now[0] = 100.0
        return {
            "disk_sampling_complete": True,
            "persistent_cache_bytes": 0,
            "peak_additional_bytes": 0,
            "disk_sample_count": 1,
        }

    monkeypatch.setattr(
        "src.video_selection.acceptance.execute_acceptance_phase.time.monotonic",
        lambda: now[0],
    )
    monkeypatch.setattr(
        "src.video_selection.acceptance.execute_acceptance_phase."
        "build_real_application",
        lambda *_args, **_kwargs: SimpleNamespace(run=lambda _configuration: None),
    )
    monkeypatch.setattr(
        "src.video_selection.acceptance.execute_acceptance_phase."
        "InternalRunController.execute",
        execute,
    )
    monkeypatch.setattr(
        "src.video_selection.acceptance.execute_acceptance_phase.DiskUsageMonitor",
        lambda **_kwargs: SimpleNamespace(
            start=lambda: None,
            stop=stop_disk_monitor,
        ),
    )
    monkeypatch.setattr(
        "src.video_selection.acceptance.execute_acceptance_phase.GpuResourceMonitor",
        lambda **_kwargs: SimpleNamespace(
            start=lambda: None,
            stop=lambda: {
                "resource_sampling_complete": True,
                "gpu_sample_count": 1,
                "gpu_sample_error_count": 0,
                "process_gpu_baseline_mib": 0,
                "system_gpu_baseline_mib": 0,
                "system_global_gpu_peak_mib": 0,
                "ollama_global_gpu_peak_mib": 0,
                "stt_global_gpu_peak_mib": 0,
                "ollama_model_size_bytes": 0,
                "ollama_model_size_vram_bytes": 0,
                "ollama_model_observed": False,
                "ollama_model_fully_resident": False,
            },
        ),
    )

    # Act
    exit_code, phase_record, report, selection_artifact = execute_acceptance_phase(
        phase="cold",
        configuration=configuration,
        resolved_models=resolved_models,
        suite_root=tmp_path / "suite",
    )

    # Assert
    assert exit_code == 1
    assert phase_record["duration_seconds"] == 10.0
    assert phase_record["resource_sampling_complete"] is True
    assert report is None
    assert selection_artifact is None
    assert now[0] == 100.0
