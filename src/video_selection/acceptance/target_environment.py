"""supported Windows 11/WSL2/Ubuntu/RTX target preflight。"""

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

from ..media.ffmpeg_media_runtime import FfmpegMediaRuntime
from .gpu_resource_monitor import find_nvidia_smi

_WINDOWS_BUILD_MINIMUM = 22000


def probe_target_environment() -> dict[str, object]:
    """supported targetを検証してprivacy-safe runtime identityを返す。"""
    if platform.system() != "Linux" or "microsoft" not in platform.release().casefold():
        raise ValueError("Target acceptanceにはWSL2 Linuxが必要です")
    os_release = _os_release()
    if os_release.get("ID") != "ubuntu" or os_release.get("VERSION_ID") != "24.04":
        raise ValueError("Target acceptanceにはUbuntu 24.04が必要です")
    windows = _windows_identity()
    windows_build = windows["build"]
    windows_edition = windows["edition"]
    if not isinstance(windows_build, int) or not isinstance(windows_edition, str):
        raise ValueError("Windows host identityが不正です")
    if windows_build < _WINDOWS_BUILD_MINIMUM or windows_edition != "Professional":
        raise ValueError("Target acceptanceにはWindows 11 Pro hostが必要です")
    gpu = _gpu_identity()
    gpu_name = gpu["name"]
    gpu_driver = gpu["driver"]
    gpu_memory_total_mib = gpu["memory_total_mib"]
    if (
        not isinstance(gpu_name, str)
        or not isinstance(gpu_driver, str)
        or not isinstance(gpu_memory_total_mib, int)
    ):
        raise ValueError("NVIDIA GPU identityが不正です")
    if "RTX 5090" not in gpu_name:
        raise ValueError("Target acceptanceにはNVIDIA GeForce RTX 5090が必要です")
    if sys.version_info < (3, 13):
        raise ValueError("Target acceptanceにはPython 3.13以上が必要です")
    media = FfmpegMediaRuntime().preflight()
    return {
        "host_os": "windows_11_pro",
        "windows_build": windows_build,
        "environment": "wsl2",
        "distribution": "ubuntu_24.04",
        "kernel": platform.release(),
        "python": platform.python_version(),
        "cpu": _cpu_model(),
        "logical_cpu_count": _logical_cpu_count(),
        "visible_ram_bytes": _visible_ram_bytes(),
        "gpu": gpu_name,
        "gpu_memory_total_mib": gpu_memory_total_mib,
        "nvidia_driver": gpu_driver,
        "ffmpeg": media.ffmpeg_version,
        "ffprobe": media.ffprobe_version,
        "media_runtime_capability_digest": media.build_capability_sha256,
    }


def probe_source_revision(repository: Path) -> tuple[str, bool]:
    """current Git commitとdirty状態を返す。"""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        raise ValueError("Acceptance source revisionを解決できません") from None
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise ValueError("Acceptance source commitが不正です")
    return commit, bool(status.strip())


def _os_release() -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        lines = Path("/etc/os-release").read_text(encoding="utf-8").splitlines()
    except OSError:
        raise ValueError("WSL distribution identityを読み込めません") from None
    for line in lines:
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip('"')
    return values


def _windows_identity() -> dict[str, object]:
    powershell = shutil.which("powershell.exe")
    if powershell is None:
        raise ValueError("Windows host identityを取得できません")
    script = (
        "$v=Get-ItemProperty 'HKLM:\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion';"
        "@{build=[int]$v.CurrentBuildNumber;edition=$v.EditionID}|"
        "ConvertTo-Json -Compress"
    )
    try:
        output = subprocess.run(
            [powershell, "-NoProfile", "-NonInteractive", "-Command", script],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        value: object = json.loads(output)
    except (OSError, subprocess.CalledProcessError, TypeError, ValueError):
        raise ValueError("Windows host identityを取得できません") from None
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("build"), int)
        or not isinstance(value.get("edition"), str)
    ):
        raise ValueError("Windows host identityが不正です")
    return value


def _gpu_identity() -> dict[str, object]:
    command = find_nvidia_smi()
    try:
        output = subprocess.run(
            [
                command,
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        raise ValueError("NVIDIA GPU identityを取得できません") from None
    parts = [part.strip() for part in output.split(",")]
    if len(parts) != 3 or not parts[2].isdigit():
        raise ValueError("NVIDIA GPU identityが不正です")
    return {"name": parts[0], "driver": parts[1], "memory_total_mib": int(parts[2])}


def _cpu_model() -> str:
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except OSError:
        return platform.processor() or "unknown"
    match = re.search(r"^model name\s*:\s*(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match is not None else "unknown"


def _logical_cpu_count() -> int:
    count = os.cpu_count()
    return 0 if count is None else count


def _visible_ram_bytes() -> int:
    try:
        text = Path("/proc/meminfo").read_text(encoding="utf-8")
    except OSError:
        return 0
    match = re.search(r"^MemTotal:\s*([0-9]+)\s*kB$", text, re.MULTILINE)
    return 0 if match is None else int(match.group(1)) * 1024
