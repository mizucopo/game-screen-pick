"""available memory取得serviceのtest。"""

import os
from pathlib import Path

import pytest

from src.video_selection.services.read_available_memory_bytes import (
    read_available_memory_bytes,
)


def test_available_memory_is_read_from_procfs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LinuxのMemAvailableがbyte数へ変換されること。

    Arrange:
        - MemAvailableを含むprocfs内容が用意される
    Act:
        - available memoryが取得される
    Assert:
        - kB値がbyte数へ変換されて返されること
    """

    # Arrange
    def read_meminfo(_path: Path, *, encoding: str) -> str:
        del encoding
        return "MemTotal: 8192 kB\nMemAvailable: 4096 kB\n"

    monkeypatch.setattr(Path, "read_text", read_meminfo)

    # Act
    actual = read_available_memory_bytes()

    # Assert
    assert actual == 4096 * 1024


def test_available_memory_falls_back_to_sysconf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """procfsを読めない環境でsysconf値が利用されること。

    Arrange:
        - procfs読込が失敗し、page sizeとavailable page数が用意される
    Act:
        - available memoryが取得される
    Assert:
        - sysconf値からbyte数が返されること
    """

    # Arrange
    def fail_read(_path: Path, *, encoding: str) -> str:
        del encoding
        raise OSError("procfs unavailable")

    values = {"SC_PAGE_SIZE": 4096, "SC_AVPHYS_PAGES": 2048}
    monkeypatch.setattr(Path, "read_text", fail_read)
    monkeypatch.setattr(os, "sysconf", values.__getitem__)

    # Act
    actual = read_available_memory_bytes()

    # Assert
    assert actual == 4096 * 2048


def test_available_memory_is_unknown_when_providers_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """すべてのresource providerが失敗した場合にNoneが返されること。

    Arrange:
        - procfs読込とsysconf取得の両方が失敗する
    Act:
        - available memoryが取得される
    Assert:
        - 安全側の不明値Noneが返されること
    """

    # Arrange
    def fail_read(_path: Path, *, encoding: str) -> str:
        del encoding
        raise OSError("procfs unavailable")

    def fail_sysconf(_name: str) -> int:
        raise ValueError("sysconf unavailable")

    monkeypatch.setattr(Path, "read_text", fail_read)
    monkeypatch.setattr(os, "sysconf", fail_sysconf)

    # Act
    actual = read_available_memory_bytes()

    # Assert
    assert actual is None
