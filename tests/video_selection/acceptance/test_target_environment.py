"""supported target environment preflightのtest。"""

import platform
import subprocess

import pytest

from src.video_selection.acceptance import target_environment
from src.video_selection.models.media_runtime_identity import MediaRuntimeIdentity


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
