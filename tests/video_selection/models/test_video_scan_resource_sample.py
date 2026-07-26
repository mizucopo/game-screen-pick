"""VideoScanResourceSampleのtest。"""

from dataclasses import replace

import pytest

from src.video_selection.models.video_scan_resource_sample import (
    VideoScanResourceSample,
)


def test_privacy_safe_resource_metrics_are_exposed_as_mapping() -> None:
    """並列判断用のresource値だけがmappingとして公開されること。

    Arrange:
        - CPU、GPU、memory、diskのresource値が用意される
    Act:
        - VideoScanResourceSampleのmappingが取得される
    Assert:
        - pathやdevice identityを含まない全resource値が返されること
    """
    # Arrange
    sample = _sample()

    # Act
    result = sample.as_mapping()

    # Assert
    assert result == {
        "cpu_percent": 20.0,
        "memory_percent": 30.0,
        "decoder_percent": 40.0,
        "gpu_percent": 25.0,
        "vram_percent": 15.0,
        "disk_busy_percent": 10.0,
        "disk_read_mib_per_second": 120.0,
        "disk_read_latency_ms": 2.5,
        "cpu_saturated_core_percent": 4.0,
    }


@pytest.mark.parametrize(
    ("change", "message"),
    [
        pytest.param({"cpu_percent": -0.1}, "cpu_percent", id="negative-percent"),
        pytest.param({"vram_percent": 100.1}, "vram_percent", id="high-percent"),
        pytest.param(
            {"disk_read_mib_per_second": float("inf")},
            "disk_read_mib_per_second",
            id="infinite-throughput",
        ),
        pytest.param(
            {"disk_read_latency_ms": -0.1},
            "disk_read_latency_ms",
            id="negative-latency",
        ),
    ],
)
def test_invalid_resource_metric_is_rejected(
    change: dict[str, float],
    message: str,
) -> None:
    """有限範囲外のresource値が拒否されること。

    Arrange:
        - 一つだけ不正なresource値が用意される
    Act:
        - VideoScanResourceSampleが構築される
    Assert:
        - 該当resource名を含むValueErrorが送出されること
    """
    # Arrange
    sample = _sample()

    # Act
    with pytest.raises(ValueError) as caught:
        replace(sample, **change)

    # Assert
    assert message in str(caught.value)


def _sample() -> VideoScanResourceSample:
    """有効なresource sampleを返す。"""
    return VideoScanResourceSample(
        cpu_percent=20.0,
        memory_percent=30.0,
        decoder_percent=40.0,
        gpu_percent=25.0,
        vram_percent=15.0,
        disk_busy_percent=10.0,
        disk_read_mib_per_second=120.0,
        disk_read_latency_ms=2.5,
        cpu_saturated_core_percent=4.0,
    )
