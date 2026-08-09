"""Suite-Owned Deletion Boundaryのtest。"""

import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from src.video_selection.acceptance.suite_owned_deletion_boundary import (
    SuiteOwnedDeletionBoundary,
)


def test_directory_removal_keeps_open_parent_when_path_is_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """検証済みancestorの差替え後も外部directoryが削除されないこと。

    Arrange:
        - suite内の削除対象と同名directoryを持つ外部treeが用意される
        - recursive deletion直前にancestorを外部へのsymlinkへ差し替える
    Act:
        - suite-owned directoryの削除が実行される
    Assert:
        - 開いていたsuite側targetだけが削除され外部treeが変更されないこと
    """
    # Arrange
    suite_root = tmp_path / "suite"
    target = suite_root / "outputs" / "cold"
    target.mkdir(parents=True)
    (target / "suite.bin").write_bytes(b"suite")
    external_root = tmp_path / "external"
    external_target = external_root / "cold"
    external_target.mkdir(parents=True)
    external_file = external_target / "external.bin"
    external_file.write_bytes(b"external")
    original_rmtree: Callable[..., None] = shutil.rmtree
    attack_performed = False

    def replace_ancestor_before_removal(
        path: str | Path,
        *args: object,
        **kwargs: object,
    ) -> None:
        nonlocal attack_performed
        if not attack_performed:
            attack_performed = True
            outputs = suite_root / "outputs"
            moved_outputs = suite_root / "outputs-before-race"
            outputs.rename(moved_outputs)
            outputs.symlink_to(external_root, target_is_directory=True)
        original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(shutil, "rmtree", replace_ancestor_before_removal)
    boundary = SuiteOwnedDeletionBoundary(suite_root)

    # Act
    boundary.remove_directory(target, "Acceptance output")

    # Assert
    assert attack_performed is True
    assert external_file.read_bytes() == b"external"
    assert not (suite_root / "outputs-before-race" / "cold").exists()
