"""GPU acceptance resource monitorのtest。"""

import os
import subprocess
from threading import Event

import pytest

from src.video_selection.acceptance import gpu_resource_monitor
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
    system_samples = iter(
        (
            {
                "system_used_mib": 100,
                "process_used_mib": 10,
            },
            {
                "system_used_mib": 12000,
                "process_used_mib": 200,
            },
            {
                "system_used_mib": 5000 + 8_000_000_000 // 1024**2,
                "process_used_mib": 5000,
            },
        )
    )
    ollama_samples = iter(
        (
            {
                "ollama_size_bytes": 0,
                "ollama_size_vram_bytes": 0,
            },
            {
                "ollama_size_bytes": 8_000_000_000,
                "ollama_size_vram_bytes": 8_000_000_000,
            },
            {
                "ollama_size_bytes": 8_000_000_000,
                "ollama_size_vram_bytes": 8_000_000_000,
            },
        )
    )
    stage = ProcessingStage.BUILD_SCENE_CATALOG
    monitor = GpuResourceMonitor(
        ollama_host="http://unused",
        stage_provider=lambda: stage,
        system_probe=lambda: next(system_samples),
        ollama_probe=lambda: next(ollama_samples),
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
    assert result["stt_non_ollama_gpu_peak_mib"] == 5000
    assert result["ollama_model_size_bytes"] == 8_000_000_000
    assert result["ollama_model_size_vram_bytes"] == 8_000_000_000
    assert result["ollama_model_observed"] is True
    assert result["ollama_model_fully_resident"] is True
    assert result["resource_sampling_complete"] is True


def test_stt_peak_excludes_concurrently_resident_ollama_vram() -> None:
    """STT Stageのsystem peakから常駐Ollama model VRAMが除外されること。

    Arrange:
        - 8 GiBのOllama modelと5 GiBの非Ollama使用量を含むsampleが用意される
    Act:
        - Context Collection Stageでresource sampleが取得される
    Assert:
        - STT peakには非Ollama分の5 GiBだけが記録されること
    """
    # Arrange
    ollama_vram_bytes = 8 * 1024**3
    system_sample = {
        "system_used_mib": 13 * 1024,
        "process_used_mib": 5 * 1024,
    }
    ollama_sample = {
        "ollama_size_bytes": ollama_vram_bytes,
        "ollama_size_vram_bytes": ollama_vram_bytes,
    }
    monitor = GpuResourceMonitor(
        ollama_host="http://unused",
        stage_provider=lambda: ProcessingStage.COLLECT_CONTEXT,
        system_probe=lambda: system_sample,
        ollama_probe=lambda: ollama_sample,
        interval_seconds=100,
    )

    # Act
    monitor.start()
    result = monitor.stop()

    # Assert
    assert result["stt_non_ollama_gpu_peak_mib"] == 5 * 1024


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
    system_sample = {
        "system_used_mib": 12000,
        "process_used_mib": 200,
    }
    ollama_sample = {
        "ollama_size_bytes": 8_000_000_000,
        "ollama_size_vram_bytes": 6_000_000_000,
    }
    monitor = GpuResourceMonitor(
        ollama_host="http://unused",
        stage_provider=lambda: ProcessingStage.BUILD_SCENE_CATALOG,
        system_probe=lambda: system_sample,
        ollama_probe=lambda: ollama_sample,
        interval_seconds=100,
    )

    # Act
    monitor.start()
    result = monitor.stop()

    # Assert
    assert result["ollama_model_observed"] is True
    assert result["ollama_model_fully_resident"] is False


def test_sampling_is_incomplete_when_background_probe_outlives_stop() -> None:
    """停止timeout後もprobeが残る場合はresource計測が不完全になること。

    Arrange:
        - baseline後のbackground probeだけが停止timeoutを超えてblockする
    Act:
        - samplerが停止される
    Assert:
        - 途中snapshotがcomplete evidenceとして扱われないこと
    """
    # Arrange
    background_started = Event()
    release_background = Event()
    call_count = 0
    system_sample = {
        "system_used_mib": 100,
        "process_used_mib": 10,
    }

    def probe() -> dict[str, int]:
        nonlocal call_count
        call_count += 1
        if call_count > 1:
            background_started.set()
            release_background.wait(timeout=1)
        return system_sample

    monitor = GpuResourceMonitor(
        ollama_host="http://unused",
        stage_provider=lambda: None,
        system_probe=probe,
        ollama_probe=lambda: {
            "ollama_size_bytes": 0,
            "ollama_size_vram_bytes": 0,
        },
        interval_seconds=0.001,
        join_timeout_seconds=0.01,
    )
    monitor.start()
    assert background_started.wait(timeout=1)

    # Act
    result = monitor.stop()
    release_background.set()

    # Assert
    assert result["resource_sampling_complete"] is False


def test_slow_ollama_probe_does_not_pause_system_gpu_sampling() -> None:
    """遅いOllama観測中もsystem GPU sampleが独立して継続されること。

    Arrange:
        - 二回目から停止待ちになるOllama probeと成功するsystem probeが用意される
    Act:
        - Ollama probeの停止待ち中にmonitorが停止される
    Assert:
        - system sampleが継続され完全なresource証拠として保持されること
        - 未完了のOllama sampleだけが別errorとして記録されること
    """
    # Arrange
    ollama_blocked = Event()
    release_ollama = Event()
    system_continued = Event()
    system_call_count = 0
    ollama_call_count = 0

    def system_probe() -> dict[str, int]:
        nonlocal system_call_count
        system_call_count += 1
        if system_call_count >= 3:
            system_continued.set()
        return {
            "system_used_mib": 12000 if system_call_count == 3 else 100,
            "process_used_mib": 10,
        }

    def ollama_probe() -> dict[str, int]:
        nonlocal ollama_call_count
        ollama_call_count += 1
        if ollama_call_count > 1:
            ollama_blocked.set()
            release_ollama.wait(timeout=1)
        return {
            "ollama_size_bytes": 8_000_000_000,
            "ollama_size_vram_bytes": 8_000_000_000,
        }

    monitor = GpuResourceMonitor(
        ollama_host="http://unused",
        stage_provider=lambda: ProcessingStage.BUILD_SCENE_CATALOG,
        system_probe=system_probe,
        ollama_probe=ollama_probe,
        interval_seconds=0.001,
        join_timeout_seconds=0.01,
    )
    monitor.start()
    assert ollama_blocked.wait(timeout=1)
    system_continued.wait(timeout=0.1)

    # Act
    result = monitor.stop()
    release_ollama.set()

    # Assert
    system_sample_count = result["gpu_sample_count"]
    ollama_sample_errors = result["ollama_sample_error_count"]
    assert isinstance(system_sample_count, int)
    assert isinstance(ollama_sample_errors, int)
    assert system_continued.is_set()
    assert system_sample_count >= 3
    assert result["gpu_sample_error_count"] == 0
    assert result["system_global_gpu_peak_mib"] == 12000
    assert result["resource_sampling_complete"] is True
    assert ollama_sample_errors >= 1


def test_transient_probe_failure_is_retried_without_invalidating_sampling() -> None:
    """一時的なGPU probe失敗が同じsample内で回復されること。

    Arrange:
        - 最初の呼び出しだけ失敗するGPU probeが用意される
    Act:
        - monitorが開始・停止される
    Assert:
        - sampleが一度だけ再試行され、resource計測が完全とされること
    """
    # Arrange
    call_count = 0
    system_sample = {
        "system_used_mib": 100,
        "process_used_mib": 10,
    }

    def probe() -> dict[str, int]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("transient probe failure")
        return system_sample

    monitor = GpuResourceMonitor(
        ollama_host="http://unused",
        stage_provider=lambda: None,
        system_probe=probe,
        ollama_probe=lambda: {
            "ollama_size_bytes": 0,
            "ollama_size_vram_bytes": 0,
        },
        interval_seconds=100,
    )

    # Act
    monitor.start()
    result = monitor.stop()

    # Assert
    assert call_count == 3
    assert result["gpu_sample_error_count"] == 0
    assert result["resource_sampling_complete"] is True


def test_unrecovered_probe_failure_invalidates_sampling() -> None:
    """二回とも失敗したGPU sampleが不完全な計測として記録されること。

    Arrange:
        - 常に失敗するGPU probeが用意される
    Act:
        - monitorが開始・停止される
    Assert:
        - 各sampleが一度だけ再試行され、resource計測が不完全とされること
    """
    # Arrange
    call_count = 0

    def probe() -> dict[str, int]:
        nonlocal call_count
        call_count += 1
        raise RuntimeError("persistent probe failure")

    monitor = GpuResourceMonitor(
        ollama_host="http://unused",
        stage_provider=lambda: None,
        system_probe=probe,
        ollama_probe=lambda: {
            "ollama_size_bytes": 0,
            "ollama_size_vram_bytes": 0,
        },
        interval_seconds=100,
    )

    # Act
    monitor.start()
    result = monitor.stop()

    # Assert
    assert call_count == 4
    assert result["gpu_sample_error_count"] == 2
    assert result["resource_sampling_complete"] is False
    assert result["ollama_sample_count"] == 1
    assert result["ollama_sample_error_count"] == 0


def test_ollama_outage_preserves_successful_system_gpu_samples() -> None:
    """Ollama観測の欠測時もsystem GPU計測が保持されること。

    Arrange:
        - 成功するsystem GPU probeと常に失敗するOllama probeが用意される
    Act:
        - monitorが開始・停止される
    Assert:
        - system sampleが完全な計測として保持されOllama欠測だけが記録されること
    """
    # Arrange
    system_sample = {
        "system_used_mib": 100,
        "process_used_mib": 10,
    }

    def unavailable_ollama() -> dict[str, int]:
        raise TimeoutError("Ollama unavailable")

    monitor = GpuResourceMonitor(
        ollama_host="http://unused",
        stage_provider=lambda: None,
        system_probe=lambda: system_sample,
        ollama_probe=unavailable_ollama,
        interval_seconds=100,
    )

    # Act
    monitor.start()
    result = monitor.stop()

    # Assert
    assert result["gpu_sample_count"] == 2
    assert result["gpu_sample_error_count"] == 0
    assert result["resource_sampling_complete"] is True
    assert result["ollama_sample_count"] == 0
    assert result["ollama_sample_error_count"] == 1
    assert result["ollama_model_observed"] is False


def test_ollama_outage_does_not_erase_observed_partial_offload() -> None:
    """Ollama欠測後も既に観測されたpartial offloadが保持されること。

    Arrange:
        - 最初にpartial offloadを返し終了時には失敗するOllama probeが用意される
    Act:
        - monitorが開始・停止される
    Assert:
        - model観測済みかつfully residentではない診断が保持されること
    """
    # Arrange
    ollama_call_count = 0

    def ollama_probe() -> dict[str, int]:
        nonlocal ollama_call_count
        ollama_call_count += 1
        if ollama_call_count > 1:
            raise TimeoutError("Ollama unavailable")
        return {
            "ollama_size_bytes": 8_000_000_000,
            "ollama_size_vram_bytes": 6_000_000_000,
        }

    monitor = GpuResourceMonitor(
        ollama_host="http://unused",
        stage_provider=lambda: ProcessingStage.BUILD_SCENE_CATALOG,
        system_probe=lambda: {
            "system_used_mib": 12000,
            "process_used_mib": 200,
        },
        ollama_probe=ollama_probe,
        interval_seconds=100,
    )

    # Act
    monitor.start()
    monitor.sample_now()
    result = monitor.stop()

    # Assert
    assert result["ollama_sample_count"] == 1
    assert result["ollama_sample_error_count"] == 1
    assert result["ollama_model_observed"] is True
    assert result["ollama_model_fully_resident"] is False
    assert result["resource_sampling_complete"] is True


def test_stt_peak_is_conservative_when_ollama_observation_is_missing() -> None:
    """Ollama欠測時のSTT peakにsystem GPU使用量が全量計上されること。

    Arrange:
        - STT stageで成功するsystem GPU probeと失敗するOllama probeが用意される
    Act:
        - monitorが開始・停止される
    Assert:
        - 推測値を差し引かずsystem GPU使用量がSTT peakとして保持されること
    """

    # Arrange
    def unavailable_ollama() -> dict[str, int]:
        raise TimeoutError("Ollama unavailable")

    monitor = GpuResourceMonitor(
        ollama_host="http://unused",
        stage_provider=lambda: ProcessingStage.COLLECT_CONTEXT,
        system_probe=lambda: {
            "system_used_mib": 5000,
            "process_used_mib": 5000,
        },
        ollama_probe=unavailable_ollama,
        interval_seconds=100,
    )

    # Act
    monitor.start()
    result = monitor.stop()

    # Assert
    assert result["stt_non_ollama_gpu_peak_mib"] == 5000
    assert result["resource_sampling_complete"] is True


def test_nvidia_smi_queries_have_a_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """nvidia-smi queryへ停止上限が設定されること。

    Arrange:
        - subprocess引数を記録するdefault GPU monitorが用意される
    Act:
        - baselineと終了時sampleが取得される
    Assert:
        - すべてのnvidia-smi queryへ正のtimeoutが渡されること
    """
    # Arrange
    timeouts: list[object] = []

    def run(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        timeouts.append(kwargs.get("timeout"))
        stdout = (
            f"{os.getpid()}, 10\n"
            if "--query-compute-apps=pid,used_memory" in command
            else "100\n"
        )
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(
        gpu_resource_monitor,
        "find_nvidia_smi",
        lambda: "nvidia-smi",
    )
    monkeypatch.setattr(
        gpu_resource_monitor,
        "_query_ollama_sizes",
        lambda _host: (0, 0),
    )
    monitor = GpuResourceMonitor(
        ollama_host="http://unused",
        stage_provider=lambda: None,
        interval_seconds=100,
    )

    # Act
    monitor.start()
    monitor.stop()

    # Assert
    assert len(timeouts) == 3
    assert all(isinstance(value, int | float) and value > 0 for value in timeouts)


def test_process_gpu_baseline_is_queried_only_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """process GPU baselineがmonitor開始時に一度だけ取得されること。

    Arrange:
        - nvidia-smi境界の呼び出し回数を記録するGPU monitorが用意される
    Act:
        - baseline後に複数回のGPU sampleが取得される
    Assert:
        - process memory queryがbaseline用に一度だけ実行されること
    """
    # Arrange
    process_query_count = 0

    def query_current_process_memory(_command: str) -> int:
        nonlocal process_query_count
        process_query_count += 1
        return 10

    monkeypatch.setattr(
        gpu_resource_monitor,
        "find_nvidia_smi",
        lambda: "nvidia-smi",
    )
    monkeypatch.setattr(
        gpu_resource_monitor,
        "_query_integer",
        lambda _command: 100,
    )
    monkeypatch.setattr(
        gpu_resource_monitor,
        "_query_current_process_memory",
        query_current_process_memory,
    )
    monkeypatch.setattr(
        gpu_resource_monitor,
        "_query_ollama_sizes",
        lambda _host: (0, 0),
    )
    monitor = GpuResourceMonitor(
        ollama_host="http://unused",
        stage_provider=lambda: None,
        interval_seconds=100,
    )

    # Act
    monitor.start()
    monitor.sample_now()
    result = monitor.stop()

    # Assert
    assert result["process_gpu_baseline_mib"] == 10
    assert process_query_count == 1
