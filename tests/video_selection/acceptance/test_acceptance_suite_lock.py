"""Acceptance Suite Lockのtest。"""

from pathlib import Path

import pytest

from src.video_selection.acceptance.acceptance_suite_lock import (
    AcceptanceSuiteLock,
)


def test_lock_rejects_overlap_and_reopens_after_release(tmp_path: Path) -> None:
    """保持中だけ後発が拒否され終了後は同じlockが再取得されること。

    Arrange:
        - 一つのsuite lock pathが用意される
    Act:
        - 先発保持中と解放後に同じlockが取得される
    Assert:
        - 保持中だけ後発が拒否され残存fileを削除せず再取得できること
    """
    # Arrange
    lock_path = tmp_path / ".locks" / "release.lock"

    # Act
    with (
        AcceptanceSuiteLock(lock_path, owned_root=tmp_path),
        pytest.raises(ValueError, match="Acceptance suiteは実行中"),
        AcceptanceSuiteLock(lock_path, owned_root=tmp_path),
    ):
        pass
    with AcceptanceSuiteLock(lock_path, owned_root=tmp_path):
        pass

    # Assert
    assert lock_path.is_file()


@pytest.mark.parametrize("symlink_component", ["lock-directory", "lock-file"])
def test_lock_rejects_symlinked_path_without_external_change(
    tmp_path: Path,
    symlink_component: str,
) -> None:
    """lock pathのsymlinkが外部fileを作成・変更せず拒否されること。

    Arrange:
        - suite所有root内のlock directoryまたはlock fileが外部へ向けられる
        - 外部treeの事前snapshotが取得される
    Act:
        - suite lockの取得が試行される
    Assert:
        - symbolic linkとして拒否され外部treeが変更されないこと
    """
    # Arrange
    owned_root = tmp_path / "artifacts"
    owned_root.mkdir()
    external_root = tmp_path / "external"
    external_root.mkdir()
    external_lock = external_root / "release.lock"
    external_lock.write_bytes(b"external-lock")
    lock_directory = owned_root / ".locks"
    lock_path = lock_directory / "release.lock"
    if symlink_component == "lock-directory":
        lock_directory.symlink_to(external_root, target_is_directory=True)
    else:
        lock_directory.mkdir()
        lock_path.symlink_to(external_lock)
    external_before = {
        path.relative_to(external_root).as_posix(): path.read_bytes()
        for path in external_root.rglob("*")
        if path.is_file()
    }

    # Act
    with (
        pytest.raises(ValueError, match="symbolic link"),
        AcceptanceSuiteLock(lock_path, owned_root=owned_root),
    ):
        pass

    # Assert
    external_after = {
        path.relative_to(external_root).as_posix(): path.read_bytes()
        for path in external_root.rglob("*")
        if path.is_file()
    }
    assert external_after == external_before
