"""processが利用できるlogical CPU数のtest。"""

import os
from pathlib import Path

import pytest

from src.video_selection.services.read_process_logical_cpu_count import (
    read_process_logical_cpu_count,
)


def test_logical_cpu_count_is_capped_by_affinity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """process affinityで利用可能なlogical CPU数が制限されること。

    Arrange:
        - processへ4 logical CPUだけが利用可能として報告される
        - applicable cgroup CPU quotaは存在しない
    Act:
        - processのlogical CPU数が取得される
    Assert:
        - host全体ではなくaffinity内の4が返されること
    """
    # Arrange
    monkeypatch.setattr(os, "process_cpu_count", lambda: 4)
    monkeypatch.setattr(
        "src.video_selection.services.read_process_logical_cpu_count."
        "resolve_cgroup_hierarchy_paths",
        lambda _controller: (),
    )

    # Act
    actual = read_process_logical_cpu_count()

    # Assert
    assert actual == 4


def test_logical_cpu_count_is_capped_by_cgroup_v2_ancestor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cgroup v2 ancestorのCPU quotaでlogical CPU数が制限されること。

    Arrange:
        - affinity上は64 CPUを利用できるprocessが用意される
        - current groupは無制限だが親groupは2 CPU相当へ制限される
    Act:
        - processのlogical CPU数が取得される
    Assert:
        - applicable hierarchyの最小quotaである2が返されること
    """
    # Arrange
    current = tmp_path / "current"
    parent = tmp_path / "parent"
    current.mkdir()
    parent.mkdir()
    (current / "cpu.max").write_text("max 100000\n", encoding="utf-8")
    (parent / "cpu.max").write_text("200000 100000\n", encoding="utf-8")
    monkeypatch.setattr(os, "process_cpu_count", lambda: 64)
    monkeypatch.setattr(
        "src.video_selection.services.read_process_logical_cpu_count."
        "resolve_cgroup_hierarchy_paths",
        lambda _controller: (("v2", (current, parent)),),
    )

    # Act
    actual = read_process_logical_cpu_count()

    # Assert
    assert actual == 2


def test_logical_cpu_count_floors_fractional_cgroup_v1_quota(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cgroup v1の小数CPU quotaが安全側の整数へ切り下げられること。

    Arrange:
        - affinity上は16 CPUを利用できるprocessが用意される
        - current groupは2.5 CPU相当へ制限される
    Act:
        - processのlogical CPU数が取得される
    Assert:
        - CPU負荷をquota内へ抑える2が返されること
    """
    # Arrange
    current = tmp_path / "current"
    current.mkdir()
    (current / "cpu.cfs_quota_us").write_text("250000\n", encoding="utf-8")
    (current / "cpu.cfs_period_us").write_text("100000\n", encoding="utf-8")
    monkeypatch.setattr(os, "process_cpu_count", lambda: 16)
    monkeypatch.setattr(
        "src.video_selection.services.read_process_logical_cpu_count."
        "resolve_cgroup_hierarchy_paths",
        lambda _controller: (("v1", (current,)),),
    )

    # Act
    actual = read_process_logical_cpu_count()

    # Assert
    assert actual == 2
