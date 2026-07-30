"""Acceptance atomic JSON操作のtest。"""

from pathlib import Path

import pytest

from src.video_selection.acceptance.atomic_json import read_json_object


def test_permission_failure_is_not_treated_as_corrupt_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """一時的な読込権限失敗が破損修復へ変換されないこと。

    Arrange:
        - valid JSON fileと読込だけを拒否するfilesystem障害が用意される
    Act:
        - Acceptance JSONが読み込まれる
    Assert:
        - PermissionErrorが保持されfile bytesも変更されないこと
    """
    # Arrange
    path = tmp_path / "state.json"
    path.write_text('{"status":"completed"}\n', encoding="utf-8")
    original_read_text = Path.read_text

    def deny_state_read(
        target: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        if target == path:
            raise PermissionError("injected permission failure")
        return original_read_text(target, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "read_text", deny_state_read)

    # Act
    # Assert
    with pytest.raises(PermissionError, match="injected permission failure"):
        read_json_object(path)
    assert original_read_text(path, encoding="utf-8") == '{"status":"completed"}\n'
