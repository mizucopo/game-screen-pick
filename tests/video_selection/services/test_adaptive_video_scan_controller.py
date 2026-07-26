"""Adaptive Video Scan Controllerのcontract test。"""

import pytest

from src.video_selection.models.video_scan_resource_sample import (
    VideoScanResourceSample,
)
from src.video_selection.services.adaptive_video_scan_controller import (
    AdaptiveVideoScanController,
)


def test_nvdec_auto_starts_six_workers_when_gpu_has_headroom() -> None:
    """NVDECとGPU余力がある24論理CPU環境で6 workerが選択されること。

    Arrange:
        - 12動画、24論理CPU、auto上限6とGPU余力のあるsampleが用意される
    Act:
        - Adaptive Video Scan Controllerが構築される
    Assert:
        - 初期worker数と実行容量に6が選択されること
    """
    # Arrange
    sample = _healthy_nvdec_sample()

    # Act
    controller = AdaptiveVideoScanController(
        video_count=12,
        configured_workers="auto",
        auto_max_workers=6,
        decode_backend="nvdec",
        logical_cpu_count=24,
        initial_resource_sample=sample,
    )

    # Assert
    assert controller.current_workers == 6
    assert controller.executor_capacity == 6
    assert controller.diagnostics["initial_workers"] == 6
    assert controller.diagnostics["mode"] == "auto"


def test_cpu_auto_keeps_the_conservative_worker_limit() -> None:
    """CPU decodeではGPU余力に関係なく保守的なworker上限が選択されること。

    Arrange:
        - 12動画、24論理CPU、auto上限6とGPU余力のあるsampleが用意される
    Act:
        - CPU decode用Controllerが構築される
    Assert:
        - 従来相当の3 workerが初期値と実行容量に選択されること
    """
    # Arrange
    sample = _healthy_nvdec_sample()

    # Act
    controller = AdaptiveVideoScanController(
        video_count=12,
        configured_workers="auto",
        auto_max_workers=6,
        decode_backend="cpu",
        logical_cpu_count=24,
        initial_resource_sample=sample,
    )

    # Assert
    assert controller.current_workers == 3
    assert controller.executor_capacity == 3


def test_fixed_worker_count_ignores_dynamic_resource_changes() -> None:
    """固定worker指定ではresource sampleによる増減が行われないこと。

    Arrange:
        - 固定4 workerと強いCPU・GPU pressure sampleが用意される
    Act:
        - scan完了境界がControllerへ通知される
    Assert:
        - worker数が4のまま維持されること
    """
    # Arrange
    controller = AdaptiveVideoScanController(
        video_count=12,
        configured_workers=4,
        auto_max_workers=6,
        decode_backend="nvdec",
        logical_cpu_count=24,
        initial_resource_sample=None,
    )

    # Act
    controller.observe_scan_completion(
        reused=False,
        input_seconds_per_wall_second=1.1,
        resource_sample=_pressure_sample(),
    )

    # Assert
    assert controller.current_workers == 4
    assert controller.executor_capacity == 4
    assert controller.diagnostics["mode"] == "fixed"


def test_auto_worker_count_changes_only_one_at_each_completion_boundary() -> None:
    """auto worker数が各scan完了境界で最大1だけ増減されること。

    Arrange:
        - GPU sample欠落により保守的な3 workerで開始するControllerが用意される
        - 続いてGPU余力sampleとCPU pressure sampleが用意される
    Act:
        - 二つのscan完了境界が順に通知される
    Assert:
        - worker数が3から4、4から3へ一つずつ変更されること
        - 変更理由と判断metricが診断へ記録されること
    """
    # Arrange
    controller = AdaptiveVideoScanController(
        video_count=12,
        configured_workers="auto",
        auto_max_workers=6,
        decode_backend="nvdec",
        logical_cpu_count=24,
        initial_resource_sample=None,
    )

    # Act
    controller.observe_scan_completion(
        reused=False,
        input_seconds_per_wall_second=1.1,
        resource_sample=_healthy_nvdec_sample(),
    )
    after_growth = controller.current_workers
    controller.observe_scan_completion(
        reused=False,
        input_seconds_per_wall_second=1.0,
        resource_sample=_pressure_sample(),
    )

    # Assert
    assert after_growth == 4
    assert controller.current_workers == 3
    changes = controller.diagnostics["changes"]
    assert isinstance(changes, list)
    assert [change["reason"] for change in changes] == [
        "gpu_headroom",
        "cpu_pressure",
    ]
    assert changes[0]["metrics"]["decoder_percent"] == 40.0


def test_missing_resource_sample_does_not_grow_above_conservative_limit() -> None:
    """resource sample失敗時にworker数が増加せず安全側へ戻されること。

    Arrange:
        - GPU余力により6 workerで開始したauto Controllerが用意される
    Act:
        - resource sampleを取得できないscan完了境界が通知される
    Assert:
        - worker数が一度に1だけ減少し、増加しないこと
    """
    # Arrange
    controller = AdaptiveVideoScanController(
        video_count=12,
        configured_workers="auto",
        auto_max_workers=6,
        decode_backend="nvdec",
        logical_cpu_count=24,
        initial_resource_sample=_healthy_nvdec_sample(),
    )

    # Act
    controller.observe_scan_completion(
        reused=False,
        input_seconds_per_wall_second=1.1,
        resource_sample=None,
    )

    # Assert
    assert controller.current_workers == 5
    changes = controller.diagnostics["changes"]
    assert isinstance(changes, list)
    assert changes[0]["reason"] == "resource_sample_unavailable"


def test_incomplete_nvdec_sample_returns_toward_conservative_limit() -> None:
    """NVIDIA metricだけ欠けたsampleでも安全側へworker数が戻されること。

    Arrange:
        - 完全なGPU余力sampleにより6 workerで開始したControllerが用意される
        - CPUとmemoryだけ取得できた不完全sampleが用意される
    Act:
        - 不完全sampleを持つscan完了境界が通知される
    Assert:
        - worker数が一度に1だけ減少し欠落理由が記録されること
    """
    # Arrange
    controller = AdaptiveVideoScanController(
        video_count=12,
        configured_workers="auto",
        auto_max_workers=6,
        decode_backend="nvdec",
        logical_cpu_count=24,
        initial_resource_sample=_healthy_nvdec_sample(),
    )
    incomplete = VideoScanResourceSample(
        cpu_percent=45.0,
        memory_percent=50.0,
        decoder_percent=None,
        gpu_percent=None,
        vram_percent=None,
        disk_busy_percent=None,
        disk_read_mib_per_second=None,
    )

    # Act
    controller.observe_scan_completion(
        reused=False,
        input_seconds_per_wall_second=1.1,
        resource_sample=incomplete,
    )

    # Assert
    assert controller.current_workers == 5
    changes = controller.diagnostics["changes"]
    assert isinstance(changes, list)
    assert changes[0]["reason"] == "resource_sample_incomplete"


def test_saturated_cpu_core_ratio_reduces_nvdec_workers() -> None:
    """aggregate CPUに余力があっても飽和coreが多ければworkerが減らされること。

    Arrange:
        - aggregate CPU 45%、飽和core 35%を示すNVDEC sampleが用意される
        - GPU余力により6 workerで開始したControllerが用意される
    Act:
        - 飽和core sampleを持つscan完了境界が通知される
    Assert:
        - worker数が5へ減りCPU core pressureが理由として記録されること
    """
    # Arrange
    controller = AdaptiveVideoScanController(
        video_count=12,
        configured_workers="auto",
        auto_max_workers=6,
        decode_backend="nvdec",
        logical_cpu_count=24,
        initial_resource_sample=_healthy_nvdec_sample(),
    )
    saturated = VideoScanResourceSample(
        cpu_percent=45.0,
        memory_percent=50.0,
        decoder_percent=55.0,
        gpu_percent=30.0,
        vram_percent=25.0,
        disk_busy_percent=45.0,
        disk_read_mib_per_second=300.0,
        cpu_saturated_core_percent=35.0,
    )

    # Act
    controller.observe_scan_completion(
        reused=False,
        input_seconds_per_wall_second=1.1,
        resource_sample=saturated,
    )

    # Assert
    assert controller.current_workers == 5
    changes = controller.diagnostics["changes"]
    assert isinstance(changes, list)
    assert changes[0]["reason"] == "cpu_core_pressure"


@pytest.mark.parametrize(
    ("configured_workers", "auto_max_workers"),
    [
        pytest.param(33, 6, id="fixed-workers"),
        pytest.param("auto", 33, id="auto-max-workers"),
    ],
)
def test_unsafe_worker_upper_bound_is_rejected(
    configured_workers: str | int,
    auto_max_workers: int,
) -> None:
    """schemaを迂回した過大なworker上限もControllerで拒否されること。

    Arrange:
        - 32を超える固定worker数またはauto上限が用意される
    Act:
        - Adaptive Video Scan Controllerの構築が試行される
    Assert:
        - executor構築前にValueErrorが返されること
    """
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="32以下"):
        AdaptiveVideoScanController(
            video_count=40,
            configured_workers=configured_workers,
            auto_max_workers=auto_max_workers,
            decode_backend="nvdec",
            logical_cpu_count=64,
            initial_resource_sample=_healthy_nvdec_sample(),
        )


def _healthy_nvdec_sample() -> VideoScanResourceSample:
    """GPU余力のあるprivacy-safeなresource sampleを返す。"""
    return VideoScanResourceSample(
        cpu_percent=45.0,
        memory_percent=50.0,
        decoder_percent=40.0,
        gpu_percent=20.0,
        vram_percent=22.0,
        disk_busy_percent=40.0,
        disk_read_mib_per_second=300.0,
    )


def _pressure_sample() -> VideoScanResourceSample:
    """CPU pressureを持つprivacy-safeなresource sampleを返す。"""
    return VideoScanResourceSample(
        cpu_percent=94.0,
        memory_percent=50.0,
        decoder_percent=55.0,
        gpu_percent=30.0,
        vram_percent=25.0,
        disk_busy_percent=45.0,
        disk_read_mib_per_second=300.0,
    )
