"""Video Set discoveryのtest。"""

from collections.abc import Iterator
from pathlib import Path

import pytest

from src.video_selection.services.discover_video_set import discover_video_set


def test_natural_order_uses_normalized_path_as_tie_breaker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同じnatural keyのvideoがrelative pathで決定的に並ぶこと。

    Arrange:
        - natural keyが等しい名前のvideoが逆順に作成される
    Act:
        - Video Setがdiscoveryされる
    Assert:
        - filesystem列挙順でなくrelative path順で返されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "clip1.mp4").write_bytes(b"video-1")
    (input_folder / "clip01.mp4").write_bytes(b"video-01")
    original_iterdir = Path.iterdir

    def reverse_tied_entries(path: Path) -> Iterator[Path]:
        if path == input_folder:
            return iter((input_folder / "clip1.mp4", input_folder / "clip01.mp4"))
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", reverse_tied_entries)

    # Act
    video_set = discover_video_set(input_folder)

    # Assert
    assert video_set.relative_paths == ("clip01.mp4", "clip1.mp4")
