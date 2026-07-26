"""supported target environment preflightのtest。"""

import platform
import subprocess

import pytest

from src.video_selection.acceptance import target_environment
from src.video_selection.models.media_runtime_identity import MediaRuntimeIdentity


def test_windows_native_ollama_is_bound_to_configured_windows_address() -> None:
    """設定endpointとWindows ollama.exe listenerの対応が受理されること。

    Arrange:
        - Windows interfaceへ解決されるhostと全interfaceのollama listenerが用意される
    Act:
        - Windows native Ollama deploymentがprobeされる
    Assert:
        - IPやpathを含まないdeployment証拠だけが返されること
    """
    # Arrange
    probed_ports: list[int] = []

    def probe(port: int) -> dict[str, object]:
        probed_ports.append(port)
        return {
            "listener_addresses": ["0.0.0.0", "::"],
            "windows_addresses": ["192.0.2.10", "2001:db8::10"],
        }

    # Act
    result = target_environment.probe_windows_native_ollama(
        "http://winpc.example:11434",
        host_address_resolver=lambda _host, _port: ("192.0.2.10",),
        windows_binding_probe=probe,
    )

    # Assert
    assert probed_ports == [11434]
    assert result == {
        "deployment": "windows_native",
        "listener_process": "ollama.exe",
    }


def test_loopback_ollama_endpoint_is_not_accepted_as_windows_binding() -> None:
    """loopback endpointがWindows nativeの証拠として受理されないこと。

    Arrange:
        - WSL local processとも区別できないloopback endpointが用意される
    Act:
        - Windows native Ollama deploymentがprobeされる
    Assert:
        - 非loopback Windows addressを要求して拒否されること
    """
    # Arrange
    endpoint = "http://localhost:11434"

    # Act / Assert
    with pytest.raises(ValueError, match="非loopback"):
        target_environment.probe_windows_native_ollama(
            endpoint,
            host_address_resolver=lambda _host, _port: ("127.0.0.1",),
            windows_binding_probe=lambda _port: {},
        )


def test_non_windows_ollama_endpoint_address_is_rejected() -> None:
    """Windows interfaceでないendpointがnative deploymentとして拒否されること。

    Arrange:
        - WSL側addressへ解決されるendpointとWindows Ollama listenerが用意される
    Act:
        - Windows native Ollama deploymentがprobeされる
    Assert:
        - endpointとWindows hostのbinding不一致として拒否されること
    """
    # Arrange
    binding = {
        "listener_addresses": ["0.0.0.0"],
        "windows_addresses": ["192.0.2.10"],
    }

    # Act / Assert
    with pytest.raises(ValueError, match="Windows host address"):
        target_environment.probe_windows_native_ollama(
            "http://wsl-service.example:11434",
            host_address_resolver=lambda _host, _port: ("198.51.100.20",),
            windows_binding_probe=lambda _port: binding,
        )


def test_rtx_5090_is_selected_from_multiple_gpu_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """複数GPUのnvidia-smi出力からRTX 5090行が選択されること。

    Arrange:
        - RTX 4090とRTX 5090を含むsupported WSL2 targetが用意される
    Act:
        - target environmentがprobeされる
    Assert:
        - RTX 5090のidentityとmemoryが返されること
    """
    # Arrange
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        platform,
        "release",
        lambda: "6.6.87.2-microsoft-standard-WSL2",
    )
    monkeypatch.setattr(
        target_environment,
        "_os_release",
        lambda: {"ID": "ubuntu", "VERSION_ID": "24.04"},
    )
    monkeypatch.setattr(
        target_environment,
        "_windows_identity",
        lambda: {"build": 26100, "edition": "Professional"},
    )
    monkeypatch.setattr(target_environment, "find_nvidia_smi", lambda: "nvidia-smi")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=(),
            returncode=0,
            stdout=(
                "NVIDIA GeForce RTX 4090, 590.1, 24564\n"
                "NVIDIA GeForce RTX 5090, 590.1, 32607\n"
            ),
        ),
    )
    monkeypatch.setattr(
        target_environment,
        "FfmpegMediaRuntime",
        lambda: _MediaRuntime(),
    )
    monkeypatch.setattr(target_environment, "_cpu_model", lambda: "Intel CPU")
    monkeypatch.setattr(target_environment, "_logical_cpu_count", lambda: 24)
    monkeypatch.setattr(
        target_environment,
        "_visible_ram_bytes",
        lambda: 64 * 1024**3,
    )

    # Act
    result = target_environment.probe_target_environment()

    # Assert
    assert result["gpu"] == "NVIDIA GeForce RTX 5090"
    assert result["gpu_memory_total_mib"] == 32607
    assert result["nvidia_driver"] == "590.1"


class _MediaRuntime:
    """test用のsupported Media Runtime。"""

    def preflight(self) -> MediaRuntimeIdentity:
        """固定Media Runtime Identityが返されること。"""
        return MediaRuntimeIdentity("7.1", "7.1", "a" * 64)
