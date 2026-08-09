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
    with AcceptanceSuiteLock(lock_path):
        with pytest.raises(ValueError, match="Acceptance suiteは実行中"):
            with AcceptanceSuiteLock(lock_path):
                pass
    with AcceptanceSuiteLock(lock_path):
        reacquired = True

    # Assert
    assert reacquired is True
    assert lock_path.is_file()
