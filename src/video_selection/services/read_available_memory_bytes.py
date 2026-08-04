"""Refinement worker budget用のavailable memoryを取得する。"""

import os
import re
from pathlib import Path, PurePosixPath

_MEM_AVAILABLE_PATTERN = re.compile(r"^MemAvailable:\s+(\d+)\s+kB$", re.MULTILINE)
_MOUNT_ESCAPE_PATTERN = re.compile(r"\\([0-7]{3})")

_CgroupMount = tuple[PurePosixPath, Path, str]


def read_available_memory_bytes() -> int | None:
    """system余力をapplicable cgroup残量で制限して返す。"""
    system_available = _read_system_available_memory_bytes()
    cgroup_available = _read_cgroup_available_memory_bytes()
    known_values = tuple(
        value for value in (system_available, cgroup_available) if value is not None
    )
    return min(known_values) if known_values else None


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
    try:
        cgroup_text = Path("/proc/self/cgroup").read_text(encoding="utf-8")
        mountinfo_text = Path("/proc/self/mountinfo").read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
    unified_path, legacy_memory_path = _parse_process_cgroups(cgroup_text)
    mounts = _parse_cgroup_mounts(
        mountinfo_text,
        unified_path=unified_path,
        legacy_memory_path=legacy_memory_path,
    )
    allowances: list[int] = []
    for root, mount_point, kind in mounts:
        process_path = unified_path if kind == "v2" else legacy_memory_path
        allowance = _read_cgroup_hierarchy_allowance(
            root,
            mount_point,
            kind,
            process_path,
        )
        if allowance is not None:
            allowances.append(allowance)
    return min(allowances) if allowances else None


def _parse_process_cgroups(
    value: str,
) -> tuple[PurePosixPath | None, PurePosixPath | None]:
    unified_path: PurePosixPath | None = None
    legacy_memory_path: PurePosixPath | None = None
    for line in value.splitlines():
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        hierarchy, controllers, path_value = fields
        path = _absolute_posix_path(path_value)
        if path is None:
            continue
        if hierarchy == "0" and not controllers:
            unified_path = path
        elif "memory" in controllers.split(","):
            legacy_memory_path = path
    return (unified_path, legacy_memory_path)


def _parse_cgroup_mounts(
    value: str,
    *,
    unified_path: PurePosixPath | None,
    legacy_memory_path: PurePosixPath | None,
) -> tuple[_CgroupMount, ...]:
    mounts: list[_CgroupMount] = []
    for line in value.splitlines():
        fields = line.split()
        try:
            separator = fields.index("-")
        except ValueError:
            continue
        if separator < 6 or len(fields) <= separator + 3:
            continue
        root = _absolute_posix_path(_decode_mount_path(fields[3]))
        mount_point_value = _decode_mount_path(fields[4])
        if root is None or not mount_point_value.startswith("/"):
            continue
        file_system = fields[separator + 1]
        super_options = set(fields[separator + 3].split(","))
        if file_system == "cgroup2" and unified_path is not None:
            mounts.append((root, Path(mount_point_value), "v2"))
        elif (
            file_system == "cgroup"
            and legacy_memory_path is not None
            and "memory" in super_options
        ):
            mounts.append((root, Path(mount_point_value), "v1"))
    return tuple(mounts)


def _read_cgroup_hierarchy_allowance(
    mount_root: PurePosixPath,
    mount_point: Path,
    kind: str,
    process_path: PurePosixPath | None,
) -> int | None:
    if process_path is None:
        return None
    if process_path == PurePosixPath("/"):
        relative_path = PurePosixPath(".")
    else:
        try:
            relative_path = process_path.relative_to(mount_root)
        except ValueError:
            return None
    if ".." in relative_path.parts:
        return None
    current = mount_point.joinpath(*relative_path.parts)
    if current != mount_point and mount_point not in current.parents:
        return None
    allowances: list[int] = []
    limit_name, usage_name = (
        ("memory.max", "memory.current")
        if kind == "v2"
        else ("memory.limit_in_bytes", "memory.usage_in_bytes")
    )
    while True:
        limit = _read_nonnegative_integer(current / limit_name)
        usage = _read_nonnegative_integer(current / usage_name)
        if limit is not None and usage is not None:
            allowances.append(max(0, limit - usage))
        if current == mount_point:
            break
        current = current.parent
    return min(allowances) if allowances else None


def _read_nonnegative_integer(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None
    if not value.isdecimal():
        return None
    return int(value)


def _absolute_posix_path(value: str) -> PurePosixPath | None:
    path = PurePosixPath(value)
    return path if path.is_absolute() else None


def _decode_mount_path(value: str) -> str:
    return _MOUNT_ESCAPE_PATTERN.sub(
        lambda matched: chr(int(matched.group(1), 8)),
        value,
    )
