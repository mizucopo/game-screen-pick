"""Video Scan resource sample安全境界のcontract test。"""

import pytest

from src.video_selection.models.video_scan_resource_sample import (
    VideoScanResourceSample,
)
from src.video_selection.services.sample_video_scan_resources_safely import (
    sample_video_scan_resources_safely,
)


@pytest.mark.parametrize("fails", [False, True])
def test_resource_sampler_failure_is_normalized_to_missing_sample(fails: bool) -> None:
    """resource samplerの失敗だけがsample欠落へ変換されること。

    Arrange:
        - sampleを返す境界または例外を送出する境界が用意される
    Act:
        - Video Scan resourceの安全な取得が実行される
    Assert:
        - 成功時はsample、失敗時はNoneが返されること
    """
    # Arrange
    sample = VideoScanResourceSample(
        cpu_percent=10.0,
        memory_percent=20.0,
        decoder_percent=30.0,
        gpu_percent=40.0,
        vram_percent=50.0,
        disk_busy_percent=60.0,
        disk_read_mib_per_second=70.0,
    )

    def sampler() -> VideoScanResourceSample:
        if fails:
            raise RuntimeError("resource probe failed")
        return sample

    # Act
    result = sample_video_scan_resources_safely(sampler)

    # Assert
    assert result is (None if fails else sample)
