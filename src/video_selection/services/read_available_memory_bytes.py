"""Refinement worker budget用のavailable memoryを取得する。"""

import os
import re
from pathlib import Path

from .resolve_cgroup_hierarchy_paths import resolve_cgroup_hierarchy_paths

_MEM_AVAILABLE_PATTERN = re.compile(r"^MemAvailable:\s+(\d+)\s+kB$", re.MULTILINE)


def read_available_memory_bytes() -> int | None:
    """system余力をapplicable cgroup残量で制限して返す。"""
    system_available = _read_system_available_memory_bytes()
    if system_available is None:
        return None
    cgroup_available = _read_cgroup_available_memory_bytes()
    return (
        system_available
        if cgroup_available is None
        else min(system_available, cgroup_available)
    )


def _read_system_available_memory_bytes() -> int | None:
    """procfsまたはsysconfからsystemのmemory余力を返す。"""
    try:
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        meminfo = None
    if meminfo is not None and (matched := _MEM_AVAILABLE_PATTERN.search(meminfo)):
        return int(matched.group(1)) * 1024
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
    except (OSError, TypeError, ValueError):
        return None
    if page_size < 1 or available_pages < 1:
        return None
    return page_size * available_pages


def _read_cgroup_available_memory_bytes() -> int | None:
    """processに適用されるcgroup階層の最小memory残量を返す。"""
    allowances: list[int] = []
    for kind, hierarchy in resolve_cgroup_hierarchy_paths("memory"):
        limit_name, usage_name = (
            ("memory.max", "memory.current")
            if kind == "v2"
            else ("memory.limit_in_bytes", "memory.usage_in_bytes")
        )
        for path in hierarchy:
            limit = _read_nonnegative_integer(path / limit_name)
            usage = _read_nonnegative_integer(path / usage_name)
            if limit is not None and usage is not None:
                allowances.append(max(0, limit - usage))
    return min(allowances) if allowances else None


def _read_nonnegative_integer(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None
    if not value.isdecimal():
        return None
    return int(value)
