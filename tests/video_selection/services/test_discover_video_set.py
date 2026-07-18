"""Video Set discoveryのtest。"""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from src.video_selection.services.discover_video_set import discover_video_set
from src.video_selection.services.validate_video_set_snapshot import (
    validate_video_set_snapshot,
)


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


def test_recursive_discovery_uses_natural_relative_path_order(tmp_path: Path) -> None:
    """recursive discoveryで相対path全体の自然順が使用されること。

    Arrange:
        - 数字を含む子directoryとvideoがfilesystem順と異なる順で作成される
    Act:
        - recursiveなVideo Set discoveryが実行される
    Assert:
        - root基準の相対path自然順でVideo Orderが確定されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    (input_folder / "chapter10").mkdir(parents=True)
    (input_folder / "chapter2").mkdir()
    (input_folder / "chapter10" / "clip1.mp4").write_bytes(b"chapter-10")
    (input_folder / "chapter2" / "clip10.mp4").write_bytes(b"chapter-2-10")
    (input_folder / "chapter2" / "clip2.mp4").write_bytes(b"chapter-2-2")

    # Act
    video_set = discover_video_set(input_folder, recursive=True)

    # Assert
    assert video_set.relative_paths == (
        "chapter2/clip2.mp4",
        "chapter2/clip10.mp4",
        "chapter10/clip1.mp4",
    )


def test_non_recursive_discovery_ignores_child_videos(tmp_path: Path) -> None:
    """recursive未指定時に子directoryのvideoが探索されないこと。

    Arrange:
        - root videoと子directory videoが用意される
    Act:
        - 既定のVideo Set discoveryが実行される
    Assert:
        - root videoだけがVideo Setへ含まれること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    child_folder = input_folder / "child"
    child_folder.mkdir(parents=True)
    (input_folder / "root.mp4").write_bytes(b"root")
    (child_folder / "child.mp4").write_bytes(b"child")

    # Act
    video_set = discover_video_set(input_folder)

    # Assert
    assert video_set.relative_paths == ("root.mp4",)


def test_directory_symlink_is_not_followed_but_file_symlink_is_accepted(
    tmp_path: Path,
) -> None:
    """directory symlinkが無視されfile symlinkの内容が採用されること。

    Arrange:
        - input外のvideo fileとvideo directoryへのsymlinkが用意される
    Act:
        - recursiveなVideo Set discoveryが実行される
    Assert:
        - file symlinkだけがVideo Sourceとして含まれること
    """
    # Arrange
    external_folder = tmp_path / "external"
    external_folder.mkdir()
    external_video = external_folder / "outside.mp4"
    external_video.write_bytes(b"outside-video")
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "linked-file.mp4").symlink_to(external_video)
    (input_folder / "linked-directory").symlink_to(
        external_folder,
        target_is_directory=True,
    )

    # Act
    video_set = discover_video_set(input_folder, recursive=True)

    # Assert
    assert video_set.relative_paths == ("linked-file.mp4",)
    assert video_set.sources[0].fingerprint == (
        discover_video_set(external_folder).sources[0].fingerprint
    )


def test_rename_and_mtime_change_preserve_content_identity(tmp_path: Path) -> None:
    """renameとmtime変更後も同じVideo Fingerprintが維持されること。

    Arrange:
        - 内容が変わらない一つのvideoが用意される
    Act:
        - mtime変更とrenameの後にそれぞれdiscoveryが実行される
    Assert:
        - Video FingerprintとVideo Set Fingerprintが維持されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    video_path = input_folder / "chapter-01.mp4"
    video_path.write_bytes(b"stable-video-content")
    initial = discover_video_set(input_folder)

    # Act
    stat = video_path.stat()
    os.utime(video_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    after_mtime = discover_video_set(input_folder)
    renamed_path = input_folder / "renamed.mp4"
    video_path.rename(renamed_path)
    after_rename = discover_video_set(input_folder)

    # Assert
    assert after_mtime.sources[0].fingerprint == initial.sources[0].fingerprint
    assert after_rename.sources[0].fingerprint == initial.sources[0].fingerprint
    assert after_mtime.fingerprint == initial.fingerprint
    assert after_rename.fingerprint == initial.fingerprint


def test_content_change_changes_video_and_video_set_identity(tmp_path: Path) -> None:
    """video内容変更時にVideoとVideo SetのFingerprintが変更されること。

    Arrange:
        - 一つのvideoから初期Video Setが発見される
    Act:
        - 同じpathの内容変更後に再度discoveryが実行される
    Assert:
        - Video FingerprintとVideo Set Fingerprintが変更されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    video_path = input_folder / "chapter.mp4"
    video_path.write_bytes(b"before")
    before = discover_video_set(input_folder)

    # Act
    video_path.write_bytes(b"after-content")
    after = discover_video_set(input_folder)

    # Assert
    assert after.sources[0].fingerprint != before.sources[0].fingerprint
    assert after.fingerprint != before.fingerprint


def test_duplicate_video_content_is_rejected_with_relative_paths(
    tmp_path: Path,
) -> None:
    """同じ内容の複数videoが相対path付きでfail-fastされること。

    Arrange:
        - 異なる相対pathに同一内容のvideoが用意される
    Act:
        - Video Set discoveryが実行される
    Assert:
        - 両方の安全な相対pathを示すDuplicate Video errorになること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "chapter-01.mp4").write_bytes(b"duplicate")
    (input_folder / "chapter-02.mp4").write_bytes(b"duplicate")

    # Act / Assert
    with pytest.raises(ValueError) as error:
        discover_video_set(input_folder)
    assert "Duplicate Video" in str(error.value)
    assert "chapter-01.mp4" in str(error.value)
    assert "chapter-02.mp4" in str(error.value)
    assert str(input_folder) not in str(error.value)


def test_snapshot_validation_rejects_video_changes_before_cache_commit(
    tmp_path: Path,
) -> None:
    """発見後のVideo Set変更がsnapshot validationで拒否されること。

    Arrange:
        - 発見済みVideo Setの一つのvideoが変更される
    Act:
        - cache commit前のsnapshot validationが実行される
    Assert:
        - Video Set snapshot変更errorが返されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    video_path = input_folder / "chapter.mp4"
    video_path.write_bytes(b"before")
    video_set = discover_video_set(input_folder)
    video_path.write_bytes(b"after-content")

    # Act / Assert
    with pytest.raises(ValueError, match="Video Set snapshotが変更されました"):
        validate_video_set_snapshot(video_set)


def test_snapshot_validation_rejects_rewrite_with_restored_mtime(
    tmp_path: Path,
) -> None:
    """size・inode・mtimeが維持された内容変更もsnapshot不一致になること。

    Arrange:
        - 発見済みvideoと同じsize・inode・mtimeを保つ別内容が用意される
    Act:
        - Video Set snapshot validationが実行される
    Assert:
        - content fingerprint不一致として変更errorが返されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    video_path = input_folder / "chapter.mp4"
    video_path.write_bytes(b"before")
    video_set = discover_video_set(input_folder)
    original_stat = video_path.stat()
    video_path.write_bytes(b"after!")
    os.utime(
        video_path,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    rewritten_stat = video_path.stat()
    assert (
        rewritten_stat.st_dev,
        rewritten_stat.st_ino,
        rewritten_stat.st_size,
        rewritten_stat.st_mtime_ns,
    ) == video_set.sources[0].stat_signature[:4]
    assert rewritten_stat.st_ctime_ns != video_set.sources[0].changed_at_ns

    # Act / Assert
    with pytest.raises(ValueError, match="Video Set snapshotが変更されました"):
        validate_video_set_snapshot(video_set)
