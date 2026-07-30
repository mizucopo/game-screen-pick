"""Input Lockのreal filesystem test。"""

from pathlib import Path

import pytest

from src.video_selection.services.input_folder_lock import InputFolderLock


def test_same_input_folder_lock_is_rejected_without_waiting(tmp_path: Path) -> None:
    """同じVideo Input Folderの二つ目のlockが即時拒否されること。

    Arrange:
        - 一つ目のInput Lockが保持される
    Act:
        - 同じinput folderの二つ目のInput Lock取得が試行される
    Assert:
        - 待機せず同時実行errorが返されcache rootは作成されないこと
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()

    # Act
    # Assert
    with (
        InputFolderLock(input_folder),
        pytest.raises(RuntimeError, match="既に実行中"),
        InputFolderLock(input_folder),
    ):
        pytest.fail("二つ目のInput Lockは取得されないこと")
    assert not (input_folder / ".game-screen-pick" / "cache").exists()


def test_different_input_folders_can_be_locked_concurrently(tmp_path: Path) -> None:
    """異なるVideo Input FolderのInput Lockが同時に取得されること。

    Arrange:
        - 二つの異なるinput folderが用意される
    Act:
        - 両方のInput Lockが同時に取得される
    Assert:
        - 互いに妨げず保持状態になること
    """
    # Arrange
    first_input = tmp_path / "first"
    second_input = tmp_path / "second"
    first_input.mkdir()
    second_input.mkdir()

    # Act
    with (
        InputFolderLock(first_input) as first_lock,
        InputFolderLock(second_input) as second_lock,
    ):
        both_held = first_lock.is_held and second_lock.is_held

    # Assert
    assert both_held
