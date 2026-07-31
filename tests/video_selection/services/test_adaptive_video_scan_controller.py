"""Adaptive Video Scan Controllerのcontract test。"""

from dataclasses import replace

import pytest

from src.video_selection.models.video_scan_resource_sample import (
    VideoScanResourceSample,
)
from src.video_selection.services.adaptive_video_scan_controller import (
    AdaptiveVideoScanController,
)


def test_nvdec_auto_grows_to_six_workers_after_rolling_gpu_headroom() -> None:
    """NVDECと継続GPU余力がある24論理CPU環境で6 workerまで増加されること。

    Arrange:
        - 12動画、24論理CPU、auto上限6とGPU余力のあるsampleが用意される
    Act:
        - Controllerが構築され6回のscan完了が通知される
    Assert:
        - 初期値3からrolling判断で6まで段階的に増加されること
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
    for _ in range(6):
        controller.observe_scan_completion(
            reused=False,
            input_seconds_per_wall_second=1.1,
            resource_sample=sample,
        )

    # Assert
    assert controller.current_workers == 6
    assert controller.executor_capacity == 6
    assert controller.diagnostics["initial_workers"] == 3
    assert controller.diagnostics["peak_workers"] == 6
    assert controller.diagnostics["mode"] == "auto"


def test_auto_does_not_count_growth_after_all_scans_completed() -> None:
    """未完了scanを投入できない増加がpeak workerへ記録されないこと。

    Arrange:
        - 初期3 workerで4動画を処理するGPU余力付きControllerが用意される
    Act:
        - 4件すべてのscan完了境界が通知される
    Assert:
        - 実際に4並列で走らないためworker数とpeakが3のままになること
    """
    # Arrange
    sample = _healthy_nvdec_sample()
    controller = AdaptiveVideoScanController(
        video_count=4,
        configured_workers="auto",
        auto_max_workers=6,
        decode_backend="nvdec",
        logical_cpu_count=24,
        initial_resource_sample=sample,
    )

    # Act
    for _ in range(4):
        controller.observe_scan_completion(
            reused=False,
            input_seconds_per_wall_second=1.1,
            resource_sample=sample,
        )

    # Assert
    assert controller.current_workers == 3
    assert controller.diagnostics["peak_workers"] == 3
    assert controller.diagnostics["changes"] == []


@pytest.mark.parametrize(
    ("video_count", "expected_workers"),
    [
        pytest.param(7, 3, id="growth-window-not-reachable"),
        pytest.param(8, 4, id="first-growth-reachable"),
        pytest.param(12, 6, id="capacity-reachable"),
    ],
)
def test_auto_reports_maximum_reachable_workers(
    video_count: int,
    expected_workers: int,
) -> None:
    """完了境界と残りscan数から到達可能な最大worker数が報告されること。

    Arrange:
        - 24論理CPUのNVDEC auto ControllerがVideo数ごとに用意される
    Act:
        - 理想的なresource余力で到達可能な最大worker数が取得される
    Assert:
        - growth windowと未完了scan数を満たすworker数だけが返されること
    """
    # Arrange
    controller = AdaptiveVideoScanController(
        video_count=video_count,
        configured_workers="auto",
        auto_max_workers=6,
        decode_backend="nvdec",
        logical_cpu_count=24,
        initial_resource_sample=None,
    )

    # Act
    reachable_workers = controller.maximum_reachable_workers

    # Assert
    assert reachable_workers == expected_workers


@pytest.mark.parametrize(
    "change",
    [
        pytest.param({"cpu_percent": 90.0}, id="cpu"),
        pytest.param(
            {"cpu_saturated_core_percent": 30.0},
            id="cpu-core",
        ),
        pytest.param({"memory_percent": 90.0}, id="memory"),
        pytest.param({"decoder_percent": 92.0}, id="decoder"),
        pytest.param({"gpu_percent": 95.0}, id="gpu"),
        pytest.param({"vram_percent": 90.0}, id="vram"),
        pytest.param({"disk_busy_percent": 92.0}, id="disk"),
        pytest.param(
            {"disk_read_latency_ms": 50.0},
            id="disk-latency",
        ),
    ],
)
def test_nvdec_auto_starts_one_worker_when_initial_resource_is_pressured(
    change: dict[str, float],
) -> None:
    """初期resourceが圧迫されているNVDEC runが1 workerで開始されること。

    Arrange:
        - 一つのresourceだけがpressure閾値へ達した初期sampleが用意される
    Act:
        - 12動画、24論理CPU、auto上限6のControllerが構築される
    Assert:
        - pressure理由にかかわらず初期投入が1 workerへ抑制されること
        - rolling判断で回復できるexecutor容量6は維持されること
    """
    # Arrange
    sample = replace(_healthy_nvdec_sample(), **change)

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
    assert controller.current_workers == 1
    assert controller.executor_capacity == 6
    assert controller.diagnostics["initial_workers"] == 1


def test_cpu_auto_keeps_the_conservative_worker_limit() -> None:
    """CPU decodeではGPU圧迫に関係なく保守的なworker上限が選択されること。

    Arrange:
        - 12動画、24論理CPU、auto上限6とGPU圧迫sampleが用意される
    Act:
        - CPU decode用Controllerが構築される
    Assert:
        - 従来相当の3 workerが初期値と実行容量に選択されること
    """
    # Arrange
    sample = replace(
        _healthy_nvdec_sample(),
        decoder_percent=100.0,
        gpu_percent=100.0,
        vram_percent=100.0,
    )

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
        - 続いて一定期間のGPU余力sampleとCPU pressure sampleが用意される
    Act:
        - 各sampleが三つのscan完了境界で順に通知される
    Assert:
        - worker数が3から4、4から3へ一つずつ変更されること
        - 変更理由、rolling判断metric、変更経過秒が診断へ記録されること
    """
    # Arrange
    clock_values = iter((100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0))
    controller = AdaptiveVideoScanController(
        video_count=12,
        configured_workers="auto",
        auto_max_workers=6,
        decode_backend="nvdec",
        logical_cpu_count=24,
        initial_resource_sample=None,
        clock=lambda: next(clock_values),
    )

    # Act
    for _ in range(4):
        controller.observe_scan_completion(
            reused=False,
            input_seconds_per_wall_second=1.1,
            resource_sample=_healthy_nvdec_sample(),
        )
    after_growth = controller.current_workers
    for _ in range(3):
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
    assert changes[0]["elapsed_seconds"] == 4.0
    assert changes[1]["elapsed_seconds"] == 7.0
    assert changes[0]["metrics"]["rolling"]["decoder_percent"] == 40.0
    assert controller.diagnostics["scan_wall_seconds"] == 7.0


def test_single_resource_spike_does_not_reduce_workers() -> None:
    """単発のresource spikeではworker数が変更されないこと。

    Arrange:
        - 継続するGPU余力sampleにより6 workerまで増加したControllerが用意される
    Act:
        - 一度だけCPU pressure sampleが通知される
    Assert:
        - rolling windowがspikeを吸収しworker数が6のまま維持されること
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
    for _ in range(6):
        controller.observe_scan_completion(
            reused=False,
            input_seconds_per_wall_second=1.1,
            resource_sample=_healthy_nvdec_sample(),
        )
    changes_before_spike = controller.diagnostics["changes"]
    assert isinstance(changes_before_spike, list)
    change_count_before_spike = len(changes_before_spike)

    # Act
    controller.observe_scan_completion(
        reused=False,
        input_seconds_per_wall_second=1.0,
        resource_sample=_pressure_sample(),
    )

    # Assert
    assert controller.current_workers == 6
    changes = controller.diagnostics["changes"]
    assert isinstance(changes, list)
    assert len(changes) == change_count_before_spike


def test_sustained_disk_throughput_slowdown_reduces_workers() -> None:
    """継続するdisk throughput低下でworker数が減らされること。

    Arrange:
        - 高いdisk throughputで6 workerまで増加したControllerが用意される
        - disk busyを伴う低throughput sampleが用意される
    Act:
        - 高throughputと低throughputがrolling windowへ順に通知される
    Assert:
        - 単発低下では変更されず継続低下後に5 workerへ減らされること
        - disk throughput低下が変更理由と判断metricへ記録されること
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
    for _ in range(6):
        controller.observe_scan_completion(
            reused=False,
            input_seconds_per_wall_second=1.1,
            resource_sample=_healthy_nvdec_sample(),
        )
    slow_disk = VideoScanResourceSample(
        cpu_percent=45.0,
        memory_percent=50.0,
        decoder_percent=40.0,
        gpu_percent=20.0,
        vram_percent=22.0,
        disk_busy_percent=80.0,
        disk_read_mib_per_second=90.0,
        disk_read_latency_ms=20.0,
    )

    # Act
    controller.observe_scan_completion(
        reused=False,
        input_seconds_per_wall_second=1.05,
        resource_sample=slow_disk,
    )
    after_spike = controller.current_workers
    controller.observe_scan_completion(
        reused=False,
        input_seconds_per_wall_second=1.0,
        resource_sample=slow_disk,
    )

    # Assert
    assert after_spike == 6
    assert controller.current_workers == 5
    changes = controller.diagnostics["changes"]
    assert isinstance(changes, list)
    assert changes[-1]["reason"] == "disk_throughput_slowdown"
    metrics = changes[-1]["metrics"]
    assert metrics["disk_throughput_ratio"] == pytest.approx(0.3)


def test_missing_resource_samples_invalidate_stale_headroom() -> None:
    """resource sample失敗中に古い余力からworker数が再増加されないこと。

    Arrange:
        - 継続するGPU余力により6 workerまで増加したauto Controllerが用意される
    Act:
        - resource sampleを取得できない4回のscan完了境界が通知される
    Assert:
        - worker数が一度に1ずつ保守値3まで減少し、古いsampleで再増加しないこと
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
    for _ in range(6):
        controller.observe_scan_completion(
            reused=False,
            input_seconds_per_wall_second=1.1,
            resource_sample=_healthy_nvdec_sample(),
        )

    # Act
    for _ in range(4):
        controller.observe_scan_completion(
            reused=False,
            input_seconds_per_wall_second=1.1,
            resource_sample=None,
        )

    # Assert
    assert controller.current_workers == 3
    changes = controller.diagnostics["changes"]
    assert isinstance(changes, list)
    assert [change["reason"] for change in changes[-3:]] == [
        "resource_sample_unavailable",
        "resource_sample_unavailable",
        "resource_sample_unavailable",
    ]


def test_auto_does_not_grow_without_disk_and_stream_speed_evidence() -> None:
    """disk観測とstream速度が欠ける間はworker数が増加されないこと。

    Arrange:
        - GPUには余力があるがdisk値を持たないsampleが用意される
    Act:
        - stream速度なしで6回のscan完了境界が通知される
    Assert:
        - 古い値やGPU値だけでは保守値3から増加されないこと
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
        decoder_percent=40.0,
        gpu_percent=20.0,
        vram_percent=22.0,
        disk_busy_percent=None,
        disk_read_mib_per_second=None,
        disk_read_latency_ms=None,
        cpu_saturated_core_percent=10.0,
    )

    # Act
    for _ in range(6):
        controller.observe_scan_completion(
            reused=False,
            input_seconds_per_wall_second=None,
            resource_sample=incomplete,
        )

    # Assert
    assert controller.current_workers == 3
    assert controller.diagnostics["changes"] == []


def test_incomplete_attempt_keeps_wall_time_before_first_completion() -> None:
    """最初のscan完了前に停止したattemptでもwall時間が保持されること。

    Arrange:
        - 開始100秒、停止112秒を返すclockとauto Controllerが用意される
    Act:
        - scan完了なしでattempt停止が通知される
    Assert:
        - 停止完了までの12秒が比較用診断へ記録されること
    """
    # Arrange
    clock_values = iter((100.0, 112.0))
    controller = AdaptiveVideoScanController(
        video_count=12,
        configured_workers="auto",
        auto_max_workers=6,
        decode_backend="nvdec",
        logical_cpu_count=24,
        initial_resource_sample=_healthy_nvdec_sample(),
        clock=lambda: next(clock_values),
    )

    # Act
    controller.finish_incomplete_attempt()

    # Assert
    assert controller.diagnostics["completed_scans"] == 0
    assert controller.diagnostics["scan_wall_seconds"] == 12.0


def test_incomplete_nvdec_sample_returns_toward_conservative_limit() -> None:
    """NVIDIA metricだけ欠けたsampleでも安全側へworker数が戻されること。

    Arrange:
        - 完全なGPU余力sampleにより6 workerまで増加したControllerが用意される
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
    for _ in range(6):
        controller.observe_scan_completion(
            reused=False,
            input_seconds_per_wall_second=1.1,
            resource_sample=_healthy_nvdec_sample(),
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
    assert changes[-1]["reason"] == "resource_sample_incomplete"


def test_saturated_cpu_core_ratio_reduces_nvdec_workers() -> None:
    """aggregate CPUに余力があっても飽和coreが多ければworkerが減らされること。

    Arrange:
        - aggregate CPU 45%、飽和core 35%を示すNVDEC sampleが用意される
        - GPU余力により6 workerまで増加したControllerが用意される
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
    for _ in range(6):
        controller.observe_scan_completion(
            reused=False,
            input_seconds_per_wall_second=1.1,
            resource_sample=_healthy_nvdec_sample(),
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
    for _ in range(3):
        controller.observe_scan_completion(
            reused=False,
            input_seconds_per_wall_second=1.1,
            resource_sample=saturated,
        )

    # Assert
    assert controller.current_workers == 5
    changes = controller.diagnostics["changes"]
    assert isinstance(changes, list)
    assert changes[-1]["reason"] == "cpu_core_pressure"


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

    # Arrange
    expected_message = "32以下"

    # Act
    with pytest.raises(ValueError) as error:
        AdaptiveVideoScanController(
            video_count=40,
            configured_workers=configured_workers,
            auto_max_workers=auto_max_workers,
            decode_backend="nvdec",
            logical_cpu_count=64,
            initial_resource_sample=_healthy_nvdec_sample(),
        )

    # Assert
    assert expected_message in str(error.value)


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
        cpu_saturated_core_percent=10.0,
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
