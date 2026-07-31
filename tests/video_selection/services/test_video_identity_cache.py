"""Video Identity cacheのtest。"""

import hashlib
import os
from pathlib import Path
from typing import NoReturn, Protocol

import pytest

from src.video_selection.services.discover_video_set import discover_video_set
from src.video_selection.services.video_identity_cache import VideoIdentityCache
from tests.video_selection.fakes.recording_run_observer import RecordingRunObserver


class _NamedDigestFile(Protocol):
    """hashlib.file_digestへ渡せる名前付きfile。"""

    name: str

    def readinto(self, buffer: bytearray, /) -> int:
        """bufferへbytesを読み込む。"""

    def readable(self) -> bool:
        """読み込み可能か返す。"""


def test_matching_source_snapshot_reuses_identity_without_reading_video(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """一致するsource snapshotからidentityが動画再読込なしで復元されること。

    Arrange:
        - whole-file SHA-256を確定したVideo Identity cacheが用意される
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
    identity_cache_root = tmp_path / "video-identities"
    cache = VideoIdentityCache(identity_cache_root)
    first = discover_video_set(input_folder, identity_cache=cache)

    def reject_digest(*_args: object, **_kwargs: object) -> NoReturn:
        raise AssertionError("cache hitで動画を再読込してはいけません")

    monkeypatch.setattr(hashlib, "file_digest", reject_digest)

    # Act
    second = discover_video_set(input_folder, identity_cache=cache)

    # Assert
    assert second.fingerprint == first.fingerprint
    cache_entry = next(identity_cache_root.glob("*.json"))
    cache_text = cache_entry.read_text()
    assert "chapter.mp4" not in cache_text
    assert str(input_folder) not in cache_text


def test_identity_resolution_reports_durable_recompute_then_reuse(
    tmp_path: Path,
) -> None:
    """Video IdentityのSHA完了量がcheckpoint fingerprint付きで通知されること。

    Arrange:
        - 一つのvideoと記録用observer付きIdentity cacheが用意される
    Act:
        - 同じVideo Setがcoldとwarmで発見される
    Assert:
        - 同じWork Unitがrecompute、reuseの順で通知されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "chapter.mp4").write_bytes(b"stable-video")
    observer = RecordingRunObserver()
    cache = VideoIdentityCache(
        tmp_path / "video-identities",
        observer=observer,
    )

    # Act
    discover_video_set(input_folder, identity_cache=cache)
    discover_video_set(input_folder, identity_cache=cache)

    # Assert
    events = tuple(
        event
        for event in observer.progress_events
        if event.work_unit_kind == "video-identity"
    )
    assert len(events) == 2
    assert events[0].work_unit_fingerprint == events[1].work_unit_fingerprint
    assert (events[0].recompute_count, events[0].reuse_count) == (1, 0)
    assert (events[1].recompute_count, events[1].reuse_count) == (0, 1)


def test_changed_inode_and_ctime_reuse_identity_when_size_and_mtime_match(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """inodeとctimeが変わってもsizeとmtime一致時にidentityが再利用されること。

    Arrange:
        - identityがcacheされたvideoが同じ内容の別inodeへ置換される
        - file sizeとmtimeが発見時の値へ保たれる
    Act:
        - whole-file digestを禁止してVideo Set discoveryが再実行される
    Assert:
        - cache済みVideo Fingerprintが再利用されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    video_path = input_folder / "chapter.mp4"
    video_path.write_bytes(b"stable")
    cache = VideoIdentityCache(tmp_path / "video-identities")
    first = discover_video_set(input_folder, identity_cache=cache)
    original_stat = video_path.stat()
    original_inode = original_stat.st_ino
    original_ctime_ns = original_stat.st_ctime_ns
    replacement = input_folder / "replacement.mp4"
    replacement.write_bytes(b"stable")
    os.utime(
        replacement,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    replacement.replace(video_path)
    assert video_path.stat().st_ino != original_inode
    assert video_path.stat().st_ctime_ns != original_ctime_ns

    def reject_digest(*_args: object, **_kwargs: object) -> NoReturn:
        raise AssertionError("inodeとctimeだけでは動画を再読込してはいけません")

    monkeypatch.setattr(hashlib, "file_digest", reject_digest)

    # Act
    second = discover_video_set(input_folder, identity_cache=cache)

    # Assert
    assert second.sources[0].fingerprint == first.sources[0].fingerprint


def test_each_identity_is_persisted_before_a_later_video_is_interrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """後続動画で中断されても確定済み動画のidentityが再利用されること。

    Arrange:
        - 2本の動画と、2本目のdigestで中断するVideo Identity cacheが用意される
    Act:
        - 中断後に同じVideo Set discoveryが再開される
    Assert:
        - 1本目は再読込されず、未完了だった2本目だけが計算されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    first_path = input_folder / "001.mp4"
    second_path = input_folder / "002.mp4"
    first_path.write_bytes(b"first-video")
    second_path.write_bytes(b"second-video")
    cache = VideoIdentityCache(tmp_path / "video-identities")
    original_file_digest = hashlib.file_digest
    first_attempt_paths: list[str] = []

    def interrupt_second_digest(
        file: _NamedDigestFile,
        digest: str,
    ) -> object:
        name = str(file.name)
        first_attempt_paths.append(name)
        if name == str(second_path):
            raise KeyboardInterrupt
        return original_file_digest(file, digest)

    monkeypatch.setattr(hashlib, "file_digest", interrupt_second_digest)
    with pytest.raises(KeyboardInterrupt):
        discover_video_set(input_folder, identity_cache=cache)
    resumed_paths: list[str] = []

    def record_resumed_digest(
        file: _NamedDigestFile,
        digest: str,
    ) -> object:
        resumed_paths.append(str(file.name))
        return original_file_digest(file, digest)

    monkeypatch.setattr(hashlib, "file_digest", record_resumed_digest)

    # Act
    video_set = discover_video_set(input_folder, identity_cache=cache)

    # Assert
    assert first_attempt_paths == [str(first_path), str(second_path)]
    assert resumed_paths == [str(second_path)]
    assert len(video_set.sources) == 2


def test_engine_version_change_recomputes_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Identity Engine version変更時に該当identityだけが再計算されること。

    Arrange:
        - engine v1で確定された1本のVideo Identityが用意される
    Act:
        - 同じsourceがengine v2で解決される
    Assert:
        - whole-file SHA-256が1回だけ再計算されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "chapter.mp4").write_bytes(b"stable-video")
    identity_cache_root = tmp_path / "video-identities"
    discover_video_set(
        input_folder,
        identity_cache=VideoIdentityCache(
            identity_cache_root,
            engine_version="video-identity-engine-v1",
        ),
    )
    original_file_digest = hashlib.file_digest
    digest_call_count = 0

    def count_digest(*args: object, **kwargs: object) -> object:
        nonlocal digest_call_count
        digest_call_count += 1
        return original_file_digest(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(hashlib, "file_digest", count_digest)

    # Act
    discover_video_set(
        input_folder,
        identity_cache=VideoIdentityCache(
            identity_cache_root,
            engine_version="video-identity-engine-v2",
        ),
    )

    # Assert
    assert digest_call_count == 1


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
    identity_cache_root = tmp_path / "video-identities"
    cache = VideoIdentityCache(identity_cache_root)
    first = discover_video_set(input_folder, identity_cache=cache)
    entry_path = next(identity_cache_root.glob("*.json"))
    entry_path.write_text("{broken", encoding="utf-8")

    # Act
    second = discover_video_set(input_folder, identity_cache=cache)

    # Assert
    assert second.fingerprint == first.fingerprint
    assert '"schema":"game-screen-pick/video-identity-cache@2.0.0"' in (
        entry_path.read_text(encoding="utf-8")
    )


def test_symlinked_identity_cache_fails_before_reading_video(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """symlinkされたidentity cacheでは長時間hashが開始されないこと。

    Arrange:
        - 有効entryを持つ外部cacheへのdirectory symlinkが用意される
    Act:
        - symlink cache付きVideo Set discoveryが実行される
    Assert:
        - 安全でないrootとして失敗し動画内容が読まれないこと
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "chapter.mp4").write_bytes(b"stable-video")
    external_cache = tmp_path / "external-cache"
    first = discover_video_set(
        input_folder,
        identity_cache=VideoIdentityCache(external_cache),
    )
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
    with pytest.raises(RuntimeError, match="安全ではありません"):
        discover_video_set(
            input_folder,
            identity_cache=VideoIdentityCache(linked_cache),
        )

    # Assert
    assert first.sources[0].fingerprint == hashlib.sha256(b"stable-video").hexdigest()
    assert digest_call_count == 0


def test_reset_cache_instance_can_resolve_identity_again(tmp_path: Path) -> None:
    """resetされた同じcache instanceでidentityが再確定されること。

    Arrange:
        - 一つのidentityを確定済みのVideo Identity cacheが用意される
    Act:
        - cacheがresetされ、同じinstanceでsourceが再解決される
    Assert:
        - cache rootが再作成され、同じfingerprintが再確定されること
    """
    # Arrange
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    video_path = input_folder / "chapter.mp4"
    video_path.write_bytes(b"stable-video")
    identity_cache_root = tmp_path / "video-identities"
    cache = VideoIdentityCache(identity_cache_root)
    first_fingerprint, _first_stat, first_reused = cache.resolve(
        input_folder,
        video_path,
    )

    # Act
    cache.reset()
    second_fingerprint, _second_stat, second_reused = cache.resolve(
        input_folder,
        video_path,
    )

    # Assert
    assert first_reused is False
    assert second_reused is False
    assert second_fingerprint == first_fingerprint
    assert len(tuple(identity_cache_root.glob("*.json"))) == 1
