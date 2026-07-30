"""supported Windows 11/WSL2/Ubuntu/RTX target preflight。"""

import ipaddress
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit

from ..media.ffmpeg_media_runtime import FfmpegMediaRuntime
from .gpu_resource_monitor import find_nvidia_smi

_WINDOWS_BUILD_MINIMUM = 22000

HostAddressResolver = Callable[[str, int], tuple[str, ...]]
WindowsOllamaBindingProbe = Callable[[int], Mapping[str, object]]
IpAddress = ipaddress.IPv4Address | ipaddress.IPv6Address


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


def probe_windows_native_ollama(
    ollama_host: str,
    *,
    host_address_resolver: HostAddressResolver | None = None,
    windows_binding_probe: WindowsOllamaBindingProbe | None = None,
) -> dict[str, object]:
    """設定endpointがWindowsのollama.exe listenerへ結び付くことを検証する。"""
    try:
        parsed = urlsplit(ollama_host)
        hostname = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError:
        raise ValueError("Ollama endpointからWindows bindingを解決できません") from None
    if parsed.scheme not in {"http", "https"} or hostname is None:
        raise ValueError("Ollama endpointからWindows bindingを解決できません")
    resolver = host_address_resolver or _resolve_host_addresses
    configured_addresses = _parse_ip_addresses(
        resolver(hostname, port),
        "Ollama endpoint address",
    )
    if not configured_addresses or any(
        address.is_loopback or address.is_unspecified
        for address in configured_addresses
    ):
        raise ValueError(
            "Target acceptanceのOllama hostにはWindowsの非loopback addressが必要です"
        )
    binding_probe = windows_binding_probe or _windows_ollama_binding
    binding = binding_probe(port)
    windows_addresses = _parse_ip_addresses(
        binding.get("windows_addresses"),
        "Windows host address",
    )
    listener_addresses = _parse_ip_addresses(
        binding.get("listener_addresses"),
        "Windows Ollama listener",
    )
    if not configured_addresses.issubset(windows_addresses):
        raise ValueError("設定Ollama endpointはWindows host addressではありません")
    if not listener_addresses or not any(
        address.is_unspecified or address in configured_addresses
        for address in listener_addresses
    ):
        raise ValueError(
            "設定Ollama endpointを所有するWindows ollama.exeが見つかりません"
        )
    return {
        "deployment": "windows_native",
        "listener_process": "ollama.exe",
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
    powershell = _find_powershell()
    script = (
        "[Console]::OutputEncoding=[Text.Encoding]::UTF8;"
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


def _find_powershell() -> str:
    """PATHまたはWSL標準mountからWindows PowerShellを返す。"""
    executable = shutil.which("powershell.exe")
    if executable is not None:
        return executable
    wsl_executable = Path(
        "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
    )
    if wsl_executable.is_file():
        return str(wsl_executable)
    raise ValueError("Windows host identityを取得できません")


def _windows_ollama_binding(port: int) -> dict[str, object]:
    script = (
        "[Console]::OutputEncoding=[Text.Encoding]::UTF8;"
        f"$port={port};"
        "$listeners=@(Get-NetTCPConnection -State Listen -LocalPort $port "
        "-ErrorAction Stop|Where-Object{"
        "(Get-Process -Id $_.OwningProcess -ErrorAction SilentlyContinue)."
        "ProcessName -ieq 'ollama'}|"
        "Select-Object -ExpandProperty LocalAddress);"
        "$addresses=@(Get-NetIPAddress -ErrorAction Stop|"
        "Where-Object{$_.AddressState -eq 'Preferred'}|"
        "Select-Object -ExpandProperty IPAddress);"
        "@{listener_addresses=$listeners;windows_addresses=$addresses}|"
        "ConvertTo-Json -Compress -Depth 3"
    )
    try:
        output = subprocess.run(
            [
                _find_powershell(),
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                script,
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        value: object = json.loads(output)
    except (OSError, subprocess.CalledProcessError, TypeError, ValueError):
        raise ValueError("Windows Ollama listenerを取得できません") from None
    if not isinstance(value, dict):
        raise ValueError("Windows Ollama listenerが不正です")
    return cast(dict[str, object], value)


def _resolve_host_addresses(hostname: str, port: int) -> tuple[str, ...]:
    try:
        addresses = {
            str(sockaddr[0])
            for _family, _type, _protocol, _canonical, sockaddr in socket.getaddrinfo(
                hostname,
                port,
                type=socket.SOCK_STREAM,
            )
        }
    except OSError:
        raise ValueError("Ollama endpoint addressを解決できません") from None
    return tuple(sorted(addresses))


def _parse_ip_addresses(value: object, label: str) -> frozenset[IpAddress]:
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, str) for item in value
    ):
        raise ValueError(f"{label}が不正です")
    addresses: set[IpAddress] = set()
    try:
        for item in value:
            addresses.add(ipaddress.ip_address(item.split("%", 1)[0]))
    except ValueError:
        raise ValueError(f"{label}が不正です") from None
    return frozenset(addresses)


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
    rows = tuple(_parse_gpu_row(line) for line in output.splitlines() if line.strip())
    if not rows:
        raise ValueError("NVIDIA GPU identityが不正です")
    if len(rows) != 1:
        raise ValueError("Target acceptanceには単一のNVIDIA GPU構成が必要です")
    return rows[0]


def _parse_gpu_row(line: str) -> dict[str, object]:
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 3 or not parts[0] or not parts[1] or not parts[2].isdigit():
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
