"""Refinement Groupの共有resource worker policyのtest。"""

from fractions import Fraction

from src.video_selection.services.resolve_refinement_group_worker_count import (
    resolve_refinement_group_worker_count,
)

_GIB = 1024**3


def test_worker_count_reserves_cpu_for_active_video_scans() -> None:
    """active Video ScanのCPU予約を除いたworker数が返されること。

    Arrange:
        - 24 logical CPUのうち2 scan分の16 CPUが予約される
        - memoryには4 Groupを処理できる余裕がある
    Act:
        - Refinement Group worker数が解決される
    Assert:
        - 残る8 CPUに対応する2 workerが返されること
    """
    # Arrange
    ranges = ((0, 10), (20, 30), (40, 50), (60, 70))

    # Act
    actual = resolve_refinement_group_worker_count(
        ranges,
        time_base=Fraction(1, 10),
        source_width=320,
        source_height=180,
        logical_cpu_count=24,
        active_scan_logical_cpu_reservation=16,
        available_memory_bytes=64 * _GIB,
    )

    # Assert
    assert actual == 2


def test_worker_count_uses_one_worker_when_scans_hold_cpu_capacity() -> None:
    """Video ScanがCPU容量を使い切る場合も1 Groupずつ進行されること。

    Arrange:
        - 16 logical CPUの全容量がactive Video Scanへ予約される
    Act:
        - Refinement Group worker数が解決される
    Assert:
        - pipelineを停止させない最小1 workerが返されること
    """
    # Arrange
    ranges = ((0, 10), (20, 30), (40, 50), (60, 70))

    # Act
    actual = resolve_refinement_group_worker_count(
        ranges,
        time_base=Fraction(1, 10),
        source_width=320,
        source_height=180,
        logical_cpu_count=16,
        active_scan_logical_cpu_reservation=16,
        available_memory_bytes=64 * _GIB,
    )

    # Assert
    assert actual == 1


def test_worker_count_caps_large_groups_by_available_memory() -> None:
    """長い高解像度Groupの並列数がavailable memoryで制限されること。

    Arrange:
        - 20秒、960x540相当のGroupが4件用意される
        - CPUには4 worker分の余裕があるがavailable memoryは16 GiBである
    Act:
        - Refinement Group worker数が解決される
    Assert:
        - RGB frame保持を同時に増幅しない1 workerが返されること
    """
    # Arrange
    ranges = ((0, 20), (30, 50), (60, 80), (90, 110))

    # Act
    actual = resolve_refinement_group_worker_count(
        ranges,
        time_base=Fraction(1),
        source_width=1920,
        source_height=1080,
        logical_cpu_count=64,
        active_scan_logical_cpu_reservation=0,
        available_memory_bytes=16 * _GIB,
    )

    # Assert
    assert actual == 1


def test_worker_count_fails_safe_when_memory_is_unknown() -> None:
    """available memoryを取得できない場合に並列増幅されないこと。

    Arrange:
        - CPUには4 worker分の余裕がある
        - available memoryが取得できない
    Act:
        - Refinement Group worker数が解決される
    Assert:
        - 安全側の1 workerが返されること
    """
    # Arrange
    ranges = ((0, 10), (20, 30), (40, 50), (60, 70))

    # Act
    actual = resolve_refinement_group_worker_count(
        ranges,
        time_base=Fraction(1, 10),
        source_width=320,
        source_height=180,
        logical_cpu_count=64,
        active_scan_logical_cpu_reservation=0,
        available_memory_bytes=None,
    )

    # Assert
    assert actual == 1
