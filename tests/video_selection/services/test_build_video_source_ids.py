"""Video Source ID構築のtest。"""

from pathlib import Path

from src.video_selection.models.video_source import VideoSource
from src.video_selection.services.build_video_source_ids import build_video_source_ids


def _source(path: str, fingerprint: str) -> VideoSource:
    return VideoSource(
        path=Path(path),
        relative_path=path,
        fingerprint=fingerprint,
        size_bytes=1,
        modified_at_ns=1,
    )


def test_unique_prefixes_use_short_video_source_ids() -> None:
    """一意なfingerprint prefixが12文字Video Source IDへ短縮されること。

    Arrange:
        - 異なる12文字prefixを持つVideo Sourceが用意される
    Act:
        - Video Source IDが構築される
    Assert:
        - 両sourceが12文字digest IDを持つこと
    """
    # Arrange
    first = _source("first.mkv", "a" * 64)
    second = _source("second.mkv", "b" * 64)

    # Act
    identifiers = build_video_source_ids((first, second))

    # Assert
    assert identifiers == {
        first.fingerprint: "vid_" + "a" * 12,
        second.fingerprint: "vid_" + "b" * 12,
    }


def test_colliding_prefixes_expand_only_affected_video_source_ids() -> None:
    """衝突したVideo Source IDだけが完全digestへ拡張されること。

    Arrange:
        - 同じ12文字prefixの2 sourceと一意な1 sourceが用意される
    Act:
        - Video Source IDが構築される
    Assert:
        - 衝突した2 sourceだけが64文字digestを持つこと
    """
    # Arrange
    first = _source("first.mkv", "123456789abc" + "a" * 52)
    second = _source("second.mkv", "123456789abc" + "b" * 52)
    third = _source("third.mkv", "c" * 64)

    # Act
    identifiers = build_video_source_ids((first, second, third))

    # Assert
    assert identifiers == {
        first.fingerprint: "vid_" + first.fingerprint,
        second.fingerprint: "vid_" + second.fingerprint,
        third.fingerprint: "vid_" + "c" * 12,
    }
