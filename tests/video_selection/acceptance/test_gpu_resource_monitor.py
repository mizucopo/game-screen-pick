"""GPU acceptance resource monitorのtest。"""

from src.video_selection.acceptance.gpu_resource_monitor import GpuResourceMonitor
from src.video_selection.models.processing_stage import ProcessingStage


def test_gpu_peaks_are_attributed_to_vision_and_stt_stages() -> None:
    """global GPU sampleがactive Stage別peakとmodel VRAMへ分離されること。

    Arrange:
        - baseline、Vision、STTの順に値を返すprobeが用意される
    Act:
        - 各active Stageでmonitorが開始・停止される
    Assert:
        - process baseline、Ollama/STT global peak、size_vramが区別されること
    """
    # Arrange
    samples = iter(
        (
            {
                "system_used_mib": 100,
                "process_used_mib": 10,
                "ollama_size_bytes": 0,
                "ollama_size_vram_bytes": 0,
            },
            {
                "system_used_mib": 12000,
                "process_used_mib": 200,
                "ollama_size_bytes": 8_000_000_000,
                "ollama_size_vram_bytes": 8_000_000_000,
            },
            {
                "system_used_mib": 5000,
                "process_used_mib": 5000,
                "ollama_size_bytes": 8_000_000_000,
                "ollama_size_vram_bytes": 8_000_000_000,
            },
        )
    )
    stage = ProcessingStage.BUILD_SCENE_CATALOG
    monitor = GpuResourceMonitor(
        ollama_host="http://unused",
        stage_provider=lambda: stage,
        probe=lambda: next(samples),
        interval_seconds=100,
    )

    # Act
    monitor.start()
    monitor.sample_now()
    stage = ProcessingStage.COLLECT_CONTEXT
    result = monitor.stop()

    # Assert
    assert result["process_gpu_baseline_mib"] == 10
    assert result["ollama_global_gpu_peak_mib"] == 12000
    assert result["stt_global_gpu_peak_mib"] == 5000
    assert result["ollama_model_size_bytes"] == 8_000_000_000
    assert result["ollama_model_size_vram_bytes"] == 8_000_000_000
    assert result["ollama_model_observed"] is True
    assert result["ollama_model_fully_resident"] is True
    assert result["resource_sampling_complete"] is True


def test_partial_ollama_offload_is_recorded_as_not_fully_resident() -> None:
    """Ollama modelの一部がCPUへoffloadされると不合格診断になること。

    Arrange:
        - model total sizeよりsize_vramが小さいGPU sampleが用意される
    Act:
        - Vision Stage中にresource monitorがsampleされる
    Assert:
        - model観測済みかつfully residentではないと記録されること
    """
    # Arrange
    sample = {
        "system_used_mib": 12000,
        "process_used_mib": 200,
        "ollama_size_bytes": 8_000_000_000,
        "ollama_size_vram_bytes": 6_000_000_000,
    }
    monitor = GpuResourceMonitor(
        ollama_host="http://unused",
        stage_provider=lambda: ProcessingStage.BUILD_SCENE_CATALOG,
        probe=lambda: sample,
        interval_seconds=100,
    )

    # Act
    monitor.start()
    result = monitor.stop()

    # Assert
    assert result["ollama_model_observed"] is True
    assert result["ollama_model_fully_resident"] is False
