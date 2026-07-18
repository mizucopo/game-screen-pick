"""Video Identity cacheのtest。"""

import hashlib
import os
from pathlib import Path
from typing import NoReturn

import pytest

from src.video_selection.services.discover_video_set import discover_video_set
from src.video_selection.services.video_identity_cache import VideoIdentityCache


def test_matching_stat_reuses_identity_without_reading_video(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """一致するstatからcontent identityが動画再読込なしで復元されること。

    Arrange:
        - whole-file SHA-256が保存されたVideo Identity cacheが用意される
    Act:
        - file digestを禁止して同じVideo Setが再発見される
    Assert:
        - 同じVideo Fingerprintが返され、cacheにpathが保存されないこと
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    video_path = input_folder / "chapter.mp4"
    video_path.write_bytes(b"stable-video")
    processing_cache = tmp_path / "cache"
    cache = VideoIdentityCache(processing_cache)
    first = discover_video_set(input_folder)
    cache.store(first.sources[0])

    def reject_digest(*_args: object, **_kwargs: object) -> NoReturn:
        raise AssertionError("cache hitで動画を再読込してはいけません")

    monkeypatch.setattr(hashlib, "file_digest", reject_digest)

    # Act
    second = discover_video_set(input_folder, identity_cache=cache)

    # Assert
    assert second.fingerprint == first.fingerprint
    cache_entry = next((processing_cache / "video-identities").glob("*.json"))
    cache_text = cache_entry.read_text()
    assert "chapter.mp4" not in cache_text
    assert str(input_folder) not in cache_text


def test_changed_ctime_invalidates_cached_identity(tmp_path: Path) -> None:
    """mtimeが復元されてもctime変更によりcontent identityが再計算されること。

    Arrange:
        - identityがcacheされたvideoが同じsizeの別内容へ書き換えられる
        - 発見時のmtimeが書き戻される
    Act:
        - cache付きVideo Set discoveryが再実行される
    Assert:
        - 変更後contentの別Video Fingerprintが返されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    video_path = input_folder / "chapter.mp4"
    video_path.write_bytes(b"before")
    cache = VideoIdentityCache(tmp_path / "cache")
    first = discover_video_set(input_folder)
    cache.store(first.sources[0])
    original_stat = video_path.stat()
    video_path.write_bytes(b"after!")
    os.utime(
        video_path,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    assert video_path.stat().st_ctime_ns != first.sources[0].changed_at_ns

    # Act
    second = discover_video_set(input_folder, identity_cache=cache)

    # Assert
    assert second.sources[0].fingerprint != first.sources[0].fingerprint


def test_malformed_entry_is_ignored(tmp_path: Path) -> None:
    """壊れたVideo Identity cache entryがcache missとして扱われること。

    Arrange:
        - 保存後にJSONが破損したVideo Identity cacheが用意される
    Act:
        - cache付きVideo Set discoveryが再実行される
    Assert:
        - 動画内容から正しいVideo Fingerprintが返されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "chapter.mp4").write_bytes(b"stable-video")
    processing_cache = tmp_path / "cache"
    cache = VideoIdentityCache(processing_cache)
    first = discover_video_set(input_folder)
    cache.store(first.sources[0])
    entry_path = next((processing_cache / "video-identities").glob("*.json"))
    entry_path.write_text("{broken", encoding="utf-8")

    # Act
    second = discover_video_set(input_folder, identity_cache=cache)

    # Assert
    assert second.fingerprint == first.fingerprint


def test_symlinked_processing_cache_is_not_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """symlinkされたprocessing cacheからidentityが読み込まれないこと。

    Arrange:
        - 有効entryを持つ外部cacheへのdirectory symlinkが用意される
    Act:
        - symlink cache付きVideo Set discoveryが実行される
    Assert:
        - cache entryでなく動画内容からSHA-256が再計算されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "chapter.mp4").write_bytes(b"stable-video")
    external_cache = tmp_path / "external-cache"
    first = discover_video_set(input_folder)
    VideoIdentityCache(external_cache).store(first.sources[0])
    linked_cache = tmp_path / "linked-cache"
    linked_cache.symlink_to(external_cache, target_is_directory=True)
    original_file_digest = hashlib.file_digest
    digest_call_count = 0

    def count_digest(*args: object, **kwargs: object) -> object:
        nonlocal digest_call_count
        digest_call_count += 1
        return original_file_digest(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(hashlib, "file_digest", count_digest)

    # Act
    second = discover_video_set(
        input_folder,
        identity_cache=VideoIdentityCache(linked_cache),
    )

    # Assert
    assert second.fingerprint == first.fingerprint
    assert digest_call_count == 1
