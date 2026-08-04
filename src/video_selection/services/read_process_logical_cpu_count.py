"""worker予算用のprocess logical CPU容量を取得する。"""

import os
from pathlib import Path

from .resolve_cgroup_hierarchy_paths import resolve_cgroup_hierarchy_paths


def read_process_logical_cpu_count() -> int:
    """affinity容量をapplicable cgroup CPU quotaで制限して返す。"""
    affinity_count = os.process_cpu_count() or os.cpu_count() or 1
    cgroup_count = _read_cgroup_logical_cpu_count()
    return affinity_count if cgroup_count is None else min(affinity_count, cgroup_count)


def _read_cgroup_logical_cpu_count() -> int | None:
    counts: list[int] = []
    for kind, hierarchy in resolve_cgroup_hierarchy_paths("cpu"):
        for path in hierarchy:
            count = (
                _read_v2_cpu_count(path) if kind == "v2" else _read_v1_cpu_count(path)
            )
            if count is not None:
                counts.append(count)
    return min(counts) if counts else None


def _read_v2_cpu_count(path: Path) -> int | None:
    value = _read_text(path / "cpu.max")
    if value is None:
        return None
    fields = value.split()
    if len(fields) != 2 or fields[0] == "max":
        return None
    return _quota_cpu_count(fields[0], fields[1])


def _read_v1_cpu_count(path: Path) -> int | None:
    quota = _read_text(path / "cpu.cfs_quota_us")
    period = _read_text(path / "cpu.cfs_period_us")
    if quota is None or period is None:
        return None
    return _quota_cpu_count(quota, period)


def _quota_cpu_count(quota_value: str, period_value: str) -> int | None:
    try:
        quota = int(quota_value)
        period = int(period_value)
    except ValueError:
        return None
    if quota < 1 or period < 1:
        return None
    return max(1, quota // period)


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None
