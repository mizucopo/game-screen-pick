"""processに適用されるcgroup filesystem hierarchyを解決する。"""

import re
from pathlib import Path, PurePosixPath
from typing import Literal, TypeAlias

_MOUNT_ESCAPE_PATTERN = re.compile(r"\\([0-7]{3})")

CgroupKind: TypeAlias = Literal["v1", "v2"]
CgroupHierarchy: TypeAlias = tuple[CgroupKind, tuple[Path, ...]]
_CgroupMount: TypeAlias = tuple[PurePosixPath, Path, CgroupKind]


def resolve_cgroup_hierarchy_paths(
    controller: str,
) -> tuple[CgroupHierarchy, ...]:
    """current groupからmount rootまでのapplicable hierarchyを返す。"""
    if not controller or any(character in controller for character in "/,:"):
        raise ValueError("cgroup controller名が不正です")
    try:
        cgroup_text = Path("/proc/self/cgroup").read_text(encoding="utf-8")
        mountinfo_text = Path("/proc/self/mountinfo").read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return ()
    unified_path, legacy_path = _parse_process_cgroups(cgroup_text, controller)
    mounts = _parse_cgroup_mounts(
        mountinfo_text,
        controller=controller,
        unified_path=unified_path,
        legacy_path=legacy_path,
    )
    hierarchies: list[CgroupHierarchy] = []
    for mount_root, mount_point, kind in mounts:
        process_path = unified_path if kind == "v2" else legacy_path
        hierarchy = _hierarchy_paths(
            mount_root,
            mount_point,
            process_path,
        )
        if hierarchy:
            hierarchies.append((kind, hierarchy))
    return tuple(hierarchies)


def _parse_process_cgroups(
    value: str,
    controller: str,
) -> tuple[PurePosixPath | None, PurePosixPath | None]:
    unified_path: PurePosixPath | None = None
    legacy_path: PurePosixPath | None = None
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
        elif controller in controllers.split(","):
            legacy_path = path
    return (unified_path, legacy_path)


def _parse_cgroup_mounts(
    value: str,
    *,
    controller: str,
    unified_path: PurePosixPath | None,
    legacy_path: PurePosixPath | None,
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
            and legacy_path is not None
            and controller in super_options
        ):
            mounts.append((root, Path(mount_point_value), "v1"))
    return tuple(mounts)


def _hierarchy_paths(
    mount_root: PurePosixPath,
    mount_point: Path,
    process_path: PurePosixPath | None,
) -> tuple[Path, ...]:
    if process_path is None:
        return ()
    if process_path == PurePosixPath("/"):
        relative_path = PurePosixPath(".")
    else:
        try:
            relative_path = process_path.relative_to(mount_root)
        except ValueError:
            return ()
    if ".." in relative_path.parts:
        return ()
    current = mount_point.joinpath(*relative_path.parts)
    if current != mount_point and mount_point not in current.parents:
        return ()
    paths: list[Path] = []
    while True:
        paths.append(current)
        if current == mount_point:
            break
        current = current.parent
    return tuple(paths)


def _absolute_posix_path(value: str) -> PurePosixPath | None:
    path = PurePosixPath(value)
    return path if path.is_absolute() else None


def _decode_mount_path(value: str) -> str:
    return _MOUNT_ESCAPE_PATTERN.sub(
        lambda matched: chr(int(matched.group(1), 8)),
        value,
    )
