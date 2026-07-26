"""Video Scan Resource Samplerのcontract test。"""

from src.video_selection.services.video_scan_resource_sampler import (
    VideoScanResourceSampler,
)


def test_linux_and_nvidia_metrics_are_sampled_without_device_identity() -> None:
    """CPU、memory、disk、NVIDIA metricがprivacy-safeに取得されること。

    Arrange:
        - 二時点のprocfs値とnvidia-smiのutilization値が用意される
    Act:
        - Resource Samplerで二回sampleされる
    Assert:
        - 差分CPU・disk値とGPU割合だけが返されdevice名やpathが含まれないこと
    """
    # Arrange
    proc_values = {
        "/proc/stat": iter(
            (
                "cpu  100 0 100 800 0 0 0 0 0 0\n"
                "cpu0 50 0 50 400 0 0 0 0 0 0\n"
                "cpu1 50 0 50 400 0 0 0 0 0 0\n",
                "cpu  150 0 150 900 0 0 0 0 0 0\n"
                "cpu0 100 0 100 400 0 0 0 0 0 0\n"
                "cpu1 50 0 50 500 0 0 0 0 0 0\n",
            )
        ),
        "/proc/meminfo": iter(
            (
                "MemTotal: 1000 kB\nMemAvailable: 400 kB\n",
                "MemTotal: 1000 kB\nMemAvailable: 350 kB\n",
            )
        ),
        "/proc/diskstats": iter(
            (
                "8 0 sda 10 0 1000 10 0 0 0 0 0 100 100\n",
                "8 0 sda 20 0 5096 20 0 0 0 0 0 300 300\n",
            )
        ),
    }
    clock_values = iter((10.0, 12.0))

    def read_proc(path: str) -> str | None:
        return next(proc_values[path])

    sampler = VideoScanResourceSampler(
        gpu_query=lambda: "40, 20, 7168, 32608\n",
        proc_reader=read_proc,
        clock=lambda: next(clock_values),
        logical_cpu_count=lambda: 24,
        load_average=lambda: (6.0, 0.0, 0.0),
    )

    # Act
    first = sampler.sample()
    second = sampler.sample()

    # Assert
    assert first is not None
    assert first.cpu_percent == 25.0
    assert first.memory_percent == 60.0
    assert first.decoder_percent == 40.0
    assert first.gpu_percent == 20.0
    assert first.vram_percent == 7168 / 32608 * 100
    assert first.disk_busy_percent is None
    assert second is not None
    assert second.cpu_percent == 50.0
    assert second.cpu_saturated_core_percent == 50.0
    assert second.memory_percent == 65.0
    assert second.disk_busy_percent == 10.0
    assert second.disk_read_mib_per_second == 1.0
    assert second.disk_read_latency_ms == 1.0
    assert "sda" not in str(second.as_mapping())


def test_unavailable_resource_sources_return_no_sample() -> None:
    """全resource sourceが利用不能な場合にsample失敗として返されること。

    Arrange:
        - procfs、load average、NVIDIA queryが利用不能なSamplerが用意される
    Act:
        - resource sampleが要求される
    Assert:
        - 例外ではなくNoneが返されること
    """
    # Arrange
    sampler = VideoScanResourceSampler(
        gpu_query=lambda: None,
        proc_reader=lambda _path: None,
        clock=lambda: 10.0,
        logical_cpu_count=lambda: None,
        load_average=lambda: None,
    )

    # Act
    sample = sampler.sample()

    # Assert
    assert sample is None
