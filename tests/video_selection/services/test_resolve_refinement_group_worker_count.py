"""Refinement Groupの共有resource worker policyのtest。"""

from fractions import Fraction

from src.video_selection.services.resolve_refinement_group_worker_count import (
    resolve_refinement_group_worker_count,
)

_GIB = 1024**3


def test_worker_count_respects_total_cpu_capacity() -> None:
    """logical CPU容量に収まるworker数が返されること。

    Arrange:
        - 8 logical CPUと4 Groupが用意される
        - memoryには4 Groupを処理できる余裕がある
    Act:
        - Refinement Group worker数が解決される
    Assert:
        - 4 logical CPUにつき1件の2 workerが返されること
    """
    # Arrange
    ranges = ((0, 10), (20, 30), (40, 50), (60, 70))

    # Act
    actual = resolve_refinement_group_worker_count(
        ranges,
        time_base=Fraction(1, 10),
        source_width=320,
        source_height=180,
        minimum_frame_delta_ts=1,
        maximum_frame_count_per_pts=1,
        logical_cpu_count=8,
        available_memory_bytes=64 * _GIB,
    )

    # Assert
    assert actual == 2


def test_worker_count_caps_groups_when_one_but_not_two_fit_memory() -> None:
    """一Groupだけがparallel memory予算へ収まる場合に1 workerになること。

    Arrange:
        - 2秒、960x540相当のGroupが4件用意される
        - 一Groupは収まるが二Groupは収まらない6 GiBのmemoryが用意される
    Act:
        - Refinement Group worker数が解決される
    Assert:
        - memory由来の1 workerが返されること
    """
    # Arrange
    ranges = ((0, 2), (10, 12), (20, 22), (30, 32))

    # Act
    actual = resolve_refinement_group_worker_count(
        ranges,
        time_base=Fraction(1),
        source_width=1920,
        source_height=1080,
        minimum_frame_delta_ts=1,
        maximum_frame_count_per_pts=1,
        logical_cpu_count=64,
        available_memory_bytes=6 * _GIB,
    )

    # Assert
    assert actual == 1


def test_worker_count_falls_back_to_one_for_oversized_group() -> None:
    """一Groupもparallel memory予算を超える場合に逐次処理へ戻ること。

    Arrange:
        - 20秒、960x540相当のGroupが4件用意される
        - CPUには4 worker分の余裕があるがparallel予算は一Group未満である
    Act:
        - Refinement Group worker数が解決される
    Assert:
        - 従来の逐次処理を維持する1 workerが返されること
    """
    # Arrange
    ranges = ((0, 20), (30, 50), (60, 80), (90, 110))

    # Act
    actual = resolve_refinement_group_worker_count(
        ranges,
        time_base=Fraction(1),
        source_width=1920,
        source_height=1080,
        minimum_frame_delta_ts=1,
        maximum_frame_count_per_pts=1,
        logical_cpu_count=64,
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
        minimum_frame_delta_ts=1,
        maximum_frame_count_per_pts=1,
        logical_cpu_count=64,
        available_memory_bytes=None,
    )

    # Assert
    assert actual == 1


def test_worker_count_accounts_for_observed_vfr_bursts() -> None:
    """240fpsを超える実測frame密度でmemoryが過小評価されないこと。

    Arrange:
        - 1ms間隔かつ同一PTS最大2frameの1秒Groupが4件用意される
        - 240fpsなら複数Groupが収まるが実測密度では一Groupも予算を超える
    Act:
        - Refinement Group worker数が解決される
    Assert:
        - 実測timingに基づく1 workerが返されること
    """
    # Arrange
    ranges = ((0, 1000), (2000, 3000), (4000, 5000), (6000, 7000))

    # Act
    actual = resolve_refinement_group_worker_count(
        ranges,
        time_base=Fraction(1, 1000),
        source_width=1920,
        source_height=1080,
        minimum_frame_delta_ts=1,
        maximum_frame_count_per_pts=2,
        logical_cpu_count=64,
        available_memory_bytes=6 * _GIB,
    )

    # Assert
    assert actual == 1


def test_worker_count_fails_safe_when_frame_timing_is_unknown() -> None:
    """frame timing hintを取得できない場合に1 workerへ抑制されること。

    Arrange:
        - CPUとmemoryには4 Groupを処理できる余裕がある
        - 実測frame timingを取得できないsourceが用意される
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
        minimum_frame_delta_ts=None,
        maximum_frame_count_per_pts=None,
        logical_cpu_count=64,
        available_memory_bytes=64 * _GIB,
    )

    # Assert
    assert actual == 1
