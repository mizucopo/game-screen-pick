"""Refinement worker budget用のavailable memoryを取得する。"""

import os
import re
from pathlib import Path

_MEM_AVAILABLE_PATTERN = re.compile(r"^MemAvailable:\s+(\d+)\s+kB$", re.MULTILINE)


def read_available_memory_bytes() -> int | None:
    """procfsまたはsysconfから現在利用可能なmemory byte数を返す。"""
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
