"""full-scale suite匿名input materializerのtest。"""

import json
import os
from fractions import Fraction
from pathlib import Path

import pytest

from src.video_selection.acceptance import (
    full_suite_materializer as full_materializer_module,
)
from src.video_selection.acceptance.acceptance_profile import AcceptanceProfile
from src.video_selection.acceptance.full_suite_materializer import (
    FullSuiteMaterializer,
)
from src.video_selection.acceptance.release_interval import ReleaseInterval
from src.video_selection.models.media_runtime_identity import MediaRuntimeIdentity


def test_full_sources_become_anonymous_symlinks_with_measured_duration(
    tmp_path: Path,
) -> None:
    """full sourceがcopyされず匿名symlinkと実測descriptorになること。

    Arrange:
        - private filenameを持つ2本のfull-scale videoが用意される
    Act:
        - full suite input viewがmaterializeされる
    Assert:
        - source順の匿名symlinkだけが作られること
        - 実測duration、count、path非依存snapshotが記録されること
        - manifestへprivate filenameとsource pathが保存されないこと
    """
    # Arrange
    profile = _profile(tmp_path)
    profile.input_root.mkdir()
    (profile.input_root / "private-chapter-01.mkv").write_bytes(b"first")
    (profile.input_root / "private-chapter-02.mp4").write_bytes(b"second")
    materializer = FullSuiteMaterializer(
        media_runtime_probe=_media_runtime,
        media_probe=lambda _path: {
            "start": Fraction(0),
            "duration": Fraction(50),
            "end": Fraction(50),
            "streams": (("video", "h264"),),
        },
    )

    # Act
    input_folder, descriptor = materializer.materialize(
        profile,
        profile.artifact_root / "full",
    )

    # Assert
    paths = sorted(input_folder.iterdir())
    assert [path.name for path in paths] == ["scenario-001.mkv", "scenario-002.mp4"]
    assert all(path.is_symlink() for path in paths)
    assert descriptor["scenario_count"] == 2
    assert descriptor["total_duration"] == {"numerator": 100, "denominator": 1}
    manifest = (input_folder.parent / "full-materialization.json").read_text(
        encoding="utf-8"
    )
    assert "private-chapter" not in manifest
    assert str(profile.input_root) not in manifest


def test_changed_full_source_requires_reset(tmp_path: Path) -> None:
    """materialize後にsource statが変わるとresumeが拒否されること。

    Arrange:
        - 確定済みfull input viewと変更後のsourceが用意される
    Act:
        - 同じprofileでresume materializeが試行される
    Assert:
        - source snapshot不一致としてresetが必要になること
    """
    # Arrange
    profile = _profile(tmp_path)
    profile.input_root.mkdir()
    first = profile.input_root / "private-chapter-01.mkv"
    second = profile.input_root / "private-chapter-02.mp4"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    materializer = FullSuiteMaterializer(
        media_runtime_probe=_media_runtime,
        media_probe=lambda _path: {
            "start": Fraction(0),
            "duration": Fraction(50),
            "end": Fraction(50),
        },
    )
    suite_root = profile.artifact_root / "full"
    materializer.materialize(profile, suite_root)
    first.write_bytes(b"changed")

    # Act / Assert
    with pytest.raises(ValueError, match="source"):
        materializer.materialize(profile, suite_root)


def test_completed_materialization_reuses_descriptor_after_media_runtime_change(
    tmp_path: Path,
) -> None:
    """Media Runtime変更後も確定済みfull descriptorが再利用されること。

    Arrange:
        - 固定Media Runtimeで確定したfull materializationが用意される
    Act:
        - 異なるFFmpeg/ffprobe build identityでresumeが試行される
    Assert:
        - materializationをやり直さず同じdescriptorが返されること
    """
    # Arrange
    profile = _profile(tmp_path)
    profile.input_root.mkdir()
    (profile.input_root / "private-chapter-01.mkv").write_bytes(b"first")
    (profile.input_root / "private-chapter-02.mp4").write_bytes(b"second")

    def probe(_path: Path) -> dict[str, object]:
        return {
            "start": Fraction(0),
            "duration": Fraction(50),
            "end": Fraction(50),
        }

    suite_root = profile.artifact_root / "full"
    _input_folder, first = FullSuiteMaterializer(
        media_probe=probe,
        media_runtime_probe=_media_runtime,
    ).materialize(profile, suite_root)
    changed_runtime_materializer = FullSuiteMaterializer(
        media_probe=probe,
        media_runtime_probe=_changed_media_runtime,
    )

    # Act
    _input_folder, second = changed_runtime_materializer.materialize(
        profile,
        suite_root,
    )

    # Assert
    assert second == first


def test_same_size_and_mtime_replacement_reuses_completed_materialization(
    tmp_path: Path,
) -> None:
    """inodeだけが変わったsourceで確定済みmaterializationが再利用されること。

    Arrange:
        - 確定済みfull materializationとsize・mtimeが同じ置換fileが用意される
    Act:
        - 元fileが別inodeへ置換された後にresume materializeが実行される
    Assert:
        - inodeを互換性条件にせず同じdescriptorが再利用されること
        - 再利用時にmedia probeが再実行されないこと
    """
    # Arrange
    profile = _profile(tmp_path)
    profile.input_root.mkdir()
    first_source = profile.input_root / "private-chapter-01.mkv"
    first_source.write_bytes(b"first")
    (profile.input_root / "private-chapter-02.mp4").write_bytes(b"second")
    suite_root = profile.artifact_root / "full"
    materializer = FullSuiteMaterializer(
        media_runtime_probe=_media_runtime,
        media_probe=lambda _path: {
            "start": Fraction(0),
            "duration": Fraction(50),
            "end": Fraction(50),
        },
    )
    _input_folder, first_descriptor = materializer.materialize(profile, suite_root)
    original_stat = first_source.stat()
    replacement = profile.input_root / "replacement.mkv"
    replacement.write_bytes(b"other")
    os.utime(
        replacement,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    replacement.replace(first_source)
    assert first_source.stat().st_ino != original_stat.st_ino

    def reject_probe(_path: Path) -> dict[str, object]:
        raise AssertionError("resumeでmedia probeが再実行されました")

    # Act
    _input_folder, resumed_descriptor = FullSuiteMaterializer(
        media_runtime_probe=_changed_media_runtime,
        media_probe=reject_probe,
    ).materialize(profile, suite_root)

    # Assert
    assert resumed_descriptor == first_descriptor


def test_nonzero_media_start_preserves_normalized_full_duration(
    tmp_path: Path,
) -> None:
    """非0 media startでも正規化済み経過時間がfull suiteへ加算されること。

    Arrange:
        - start 5秒、経過50秒、end timestamp 55秒の2動画が用意される
    Act:
        - 100秒を期待するfull suiteがmaterializeされる
    Assert:
        - 各動画50秒として合計100秒がdescriptorへ記録されること
    """
    # Arrange
    profile = _profile(tmp_path)
    profile.input_root.mkdir()
    (profile.input_root / "private-chapter-01.mkv").write_bytes(b"first")
    (profile.input_root / "private-chapter-02.mp4").write_bytes(b"second")
    materializer = FullSuiteMaterializer(
        media_runtime_probe=_media_runtime,
        media_probe=lambda _path: {
            "start": Fraction(5),
            "duration": Fraction(50),
            "end": Fraction(55),
        },
    )

    # Act
    _, descriptor = materializer.materialize(
        profile,
        profile.artifact_root / "full",
    )

    # Assert
    assert descriptor["total_duration"] == {"numerator": 100, "denominator": 1}


def test_repointed_anonymous_symlink_requires_reset(tmp_path: Path) -> None:
    """匿名symlinkが別sourceへ付け替えられるとresumeが拒否されること。

    Arrange:
        - 確定済みfull input viewの先頭symlinkが2本目のsourceへ付け替えられる
    Act:
        - 同じprofileでresume materializeが試行される
    Assert:
        - 現在のsource対応不一致としてresetが必要になること
    """
    # Arrange
    profile = _profile(tmp_path)
    profile.input_root.mkdir()
    first = profile.input_root / "private-chapter-01.mkv"
    second = profile.input_root / "private-chapter-02.mp4"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    materializer = FullSuiteMaterializer(
        media_runtime_probe=_media_runtime,
        media_probe=lambda _path: {
            "start": Fraction(0),
            "duration": Fraction(50),
            "end": Fraction(50),
        },
    )
    suite_root = profile.artifact_root / "full"
    input_folder, _ = materializer.materialize(profile, suite_root)
    anonymous_first = input_folder / "scenario-001.mkv"
    anonymous_first.unlink()
    anonymous_first.symlink_to(second.resolve(strict=True))

    # Act
    with pytest.raises(ValueError) as error:
        materializer.materialize(profile, suite_root)

    # Assert
    assert "匿名input" in str(error.value)


def test_stray_supported_video_requires_reset(tmp_path: Path) -> None:
    """manifest外の対応videoが匿名inputに残るとresumeが拒否されること。

    Arrange:
        - 確定済みfull input viewへ余分なscenario videoが追加される
    Act:
        - 同じprofileでresume materializeが試行される
    Assert:
        - 匿名inputの完全一致違反としてresetが必要になること
    """
    # Arrange
    profile = _profile(tmp_path)
    profile.input_root.mkdir()
    (profile.input_root / "private-chapter-01.mkv").write_bytes(b"first")
    (profile.input_root / "private-chapter-02.mp4").write_bytes(b"second")
    materializer = FullSuiteMaterializer(
        media_runtime_probe=_media_runtime,
        media_probe=lambda _path: {
            "start": Fraction(0),
            "duration": Fraction(50),
            "end": Fraction(50),
        },
    )
    suite_root = profile.artifact_root / "full"
    input_folder, _ = materializer.materialize(profile, suite_root)
    (input_folder / "scenario-999.mkv").write_bytes(b"stray")

    # Act
    with pytest.raises(ValueError) as error:
        materializer.materialize(profile, suite_root)

    # Assert
    assert "匿名input" in str(error.value)


def test_completed_source_probe_survives_later_source_failure(
    tmp_path: Path,
) -> None:
    """後続sourceのprobe失敗後も完了済みsource単位が再利用されること。

    Arrange:
        - 2本目だけ初回に失敗するduration probeが用意される
    Act:
        - 初回失敗後に同じfull materializationが再実行される
    Assert:
        - 1本目は再probeされず2本目だけが再実行されること
        - 最終descriptorが2本のcheckpointから確定されること
    """
    # Arrange
    profile = _profile(tmp_path)
    profile.input_root.mkdir()
    first = profile.input_root / "private-chapter-01.mkv"
    second = profile.input_root / "private-chapter-02.mp4"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    calls: list[str] = []
    second_failed = False

    def probe(path: Path) -> dict[str, object]:
        nonlocal second_failed
        calls.append(path.name)
        if path == second and not second_failed:
            second_failed = True
            raise OSError("simulated second source failure")
        return {
            "start": Fraction(0),
            "duration": Fraction(50),
            "end": Fraction(50),
        }

    materializer = FullSuiteMaterializer(
        media_runtime_probe=_media_runtime,
        media_probe=probe,
    )
    suite_root = profile.artifact_root / "full"
    with pytest.raises(OSError, match="simulated second source failure"):
        materializer.materialize(profile, suite_root)

    # Act
    input_folder, descriptor = materializer.materialize(profile, suite_root)

    # Assert
    assert calls == [
        "private-chapter-01.mkv",
        "private-chapter-02.mp4",
        "private-chapter-02.mp4",
    ]
    assert descriptor["scenario_count"] == 2
    assert descriptor["total_duration"] == {
        "numerator": 100,
        "denominator": 1,
    }
    assert (input_folder / "scenario-001.mkv").is_symlink()


def test_media_runtime_change_restarts_only_incomplete_full_materialization(
    tmp_path: Path,
) -> None:
    """未完成full materializationが異なるMedia Runtimeを混在させないこと。

    Arrange:
        - 旧Media Runtimeで1本目が確定し2本目のduration probeだけ失敗する
    Act:
        - 新Media Runtimeで同じfull materializationが再開される
    Assert:
        - materialization内の1本目と2本目が新runtimeで再probeされること
        - pipeline cacheやsuite全体のresetなしでdescriptorが確定されること
    """
    # Arrange
    profile = _profile(tmp_path)
    profile.input_root.mkdir()
    first = profile.input_root / "private-chapter-01.mkv"
    second = profile.input_root / "private-chapter-02.mp4"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    calls: list[str] = []
    second_failed = False

    def probe(path: Path) -> dict[str, object]:
        nonlocal second_failed
        calls.append(path.name)
        if path == second and not second_failed:
            second_failed = True
            raise OSError("simulated second source failure")
        return {
            "start": Fraction(0),
            "duration": Fraction(50),
            "end": Fraction(50),
        }

    suite_root = profile.artifact_root / "full"
    with pytest.raises(OSError, match="simulated second source failure"):
        FullSuiteMaterializer(
            media_runtime_probe=_media_runtime,
            media_probe=probe,
        ).materialize(profile, suite_root)

    # Act
    _input_folder, descriptor = FullSuiteMaterializer(
        media_runtime_probe=_changed_media_runtime,
        media_probe=probe,
    ).materialize(profile, suite_root)

    # Assert
    assert calls == [
        "private-chapter-01.mkv",
        "private-chapter-02.mp4",
        "private-chapter-01.mkv",
        "private-chapter-02.mp4",
    ]
    assert descriptor["scenario_count"] == 2


def test_duration_mismatch_preserves_completed_source_checkpoints(
    tmp_path: Path,
) -> None:
    """full duration preflight failureでも確定済みsourceが保持されること。

    Arrange:
        - profile期待値と異なるdurationを返す2本のsourceが用意される
    Act:
        - full suite materializationが試行される
    Assert:
        - preflight failureとなること
        - 再開可能な匿名symlinkとsource checkpointが保持されること
    """
    # Arrange
    profile = _profile(tmp_path)
    profile.input_root.mkdir()
    (profile.input_root / "private-chapter-01.mkv").write_bytes(b"first")
    (profile.input_root / "private-chapter-02.mp4").write_bytes(b"second")
    suite_root = profile.artifact_root / "full"
    materializer = FullSuiteMaterializer(
        media_runtime_probe=_media_runtime,
        media_probe=lambda _path: {
            "start": Fraction(0),
            "duration": Fraction(40),
            "end": Fraction(40),
        },
    )

    # Act / Assert
    with pytest.raises(ValueError, match="duration"):
        materializer.materialize(profile, suite_root)
    input_folder = suite_root / "work" / "input"
    assert sorted(path.name for path in input_folder.iterdir()) == [
        "scenario-001.mkv",
        "scenario-002.mp4",
    ]
    assert len(tuple((suite_root / "work" / "source-checkpoints").glob("*.json"))) == 2


def test_source_changed_during_probe_preserves_completed_source_checkpoints(
    tmp_path: Path,
) -> None:
    """probe中のsource変更後も確定単位を保持して次回に整合させること。

    Arrange:
        - 最初のduration probe中に2本目のsourceを書き換えるprobeが用意される
    Act:
        - full suite materializationが失敗後に再実行される
    Assert:
        - source snapshot不一致として拒否されること
        - source単位checkpointが保持され再probeなしで完了すること
    """
    # Arrange
    profile = _profile(tmp_path)
    profile.input_root.mkdir()
    first = profile.input_root / "private-chapter-01.mkv"
    second = profile.input_root / "private-chapter-02.mp4"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    probe_count = 0

    def probe(_path: Path) -> dict[str, object]:
        nonlocal probe_count
        probe_count += 1
        if probe_count == 1:
            second.write_bytes(b"changed-second")
        return {
            "start": Fraction(0),
            "duration": Fraction(50),
            "end": Fraction(50),
        }

    suite_root = profile.artifact_root / "full"
    materializer = FullSuiteMaterializer(
        media_probe=probe,
        media_runtime_probe=_media_runtime,
    )

    # Act
    with pytest.raises(ValueError, match="source"):
        materializer.materialize(profile, suite_root)
    resumed_input, descriptor = materializer.materialize(profile, suite_root)

    # Assert
    assert probe_count == 2
    assert resumed_input.is_dir()
    assert descriptor["scenario_count"] == 2
    assert (suite_root / "work" / "full-materialization.json").is_file()


def test_corrupted_terminal_manifest_is_rebuilt_from_source_checkpoints(
    tmp_path: Path,
) -> None:
    """破損した終端manifestが確定sourceを再probeせず修復されること。

    Arrange:
        - 確定済み匿名source viewとscenario countだけを壊したmanifestが用意される
    Act:
        - 同じprofileでmaterializationが再開される
    Assert:
        - sourceを再probeせず同じdescriptorとsymlink対応が復元されること
        - 終端manifestの導出値が正しい値へ書き直されること
    """
    # Arrange
    profile = _profile(tmp_path)
    profile.input_root.mkdir()
    first = profile.input_root / "private-chapter-01.mkv"
    second = profile.input_root / "private-chapter-02.mp4"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    probe_count = 0

    def probe(_path: Path) -> dict[str, object]:
        nonlocal probe_count
        probe_count += 1
        return {
            "start": Fraction(0),
            "duration": Fraction(50),
            "end": Fraction(50),
        }

    suite_root = profile.artifact_root / "full"
    materializer = FullSuiteMaterializer(
        media_probe=probe,
        media_runtime_probe=_media_runtime,
    )
    input_folder, expected = materializer.materialize(profile, suite_root)
    manifest_path = input_folder.parent / "full-materialization.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["descriptor"]["scenario_count"] = 999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def unexpected_probe(_path: Path) -> dict[str, object]:
        raise AssertionError("source must not be probed during terminal recovery")

    def unexpected_runtime_probe() -> MediaRuntimeIdentity:
        raise AssertionError("current runtime must not be probed during recovery")

    # Act
    _input_folder, resumed = FullSuiteMaterializer(
        media_probe=unexpected_probe,
        media_runtime_probe=unexpected_runtime_probe,
    ).materialize(profile, suite_root)

    # Assert
    assert probe_count == 2
    assert resumed == expected
    assert (input_folder / "scenario-001.mkv").resolve(strict=True) == first
    assert (input_folder / "scenario-002.mp4").resolve(strict=True) == second
    repaired = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert repaired["descriptor"]["scenario_count"] == 2


def test_pending_source_checkpoint_recovers_after_commit_interruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """symlink置換後のcommit中断から再probeなしで復元されること。

    Arrange:
        - 検証済みsource symlinkの固定名置換後にcheckpoint昇格だけが失敗される
        - symlinkとdurable pending checkpointが残される
    Act:
        - 同じmaterializationが新しいprocessとして再開される
    Assert:
        - source probeと現在runtime probeなしでpendingが確定されること
        - 同じsymlink対応とdescriptorから終端manifestが生成されること
    """
    # Arrange
    profile = _profile(tmp_path)
    profile.input_root.mkdir()
    first = profile.input_root / "private-chapter-01.mkv"
    second = profile.input_root / "private-chapter-02.mp4"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    original_promoter = full_materializer_module._promote_checkpoint
    promotion_count = 0

    def interrupt_second_promotion(_pending: Path, _checkpoint: Path) -> None:
        nonlocal promotion_count
        promotion_count += 1
        if promotion_count == 2:
            raise OSError("injected checkpoint commit interruption")
        original_promoter(_pending, _checkpoint)

    monkeypatch.setattr(
        full_materializer_module,
        "_promote_checkpoint",
        interrupt_second_promotion,
    )
    suite_root = profile.artifact_root / "full"
    with pytest.raises(OSError, match="checkpoint commit interruption"):
        FullSuiteMaterializer(
            media_probe=lambda _path: {
                "start": Fraction(0),
                "duration": Fraction(50),
                "end": Fraction(50),
            },
            media_runtime_probe=_media_runtime,
        ).materialize(profile, suite_root)
    work_root = suite_root / "work"
    second_input = work_root / "input" / "scenario-002.mp4"
    pending_path = work_root / "source-checkpoints" / ".scenario-002.pending.json"
    checkpoint_path = work_root / "source-checkpoints" / ".scenario-002.checkpoint.json"
    assert second_input.resolve(strict=True) == second
    assert pending_path.is_file()
    assert not checkpoint_path.exists()
    monkeypatch.setattr(
        full_materializer_module,
        "_promote_checkpoint",
        original_promoter,
    )

    def unexpected_probe(_path: Path) -> dict[str, object]:
        raise AssertionError("source must not be probed during pending recovery")

    def unexpected_runtime_probe() -> MediaRuntimeIdentity:
        raise AssertionError("current runtime must not be probed during recovery")

    # Act
    input_folder, descriptor = FullSuiteMaterializer(
        media_probe=unexpected_probe,
        media_runtime_probe=unexpected_runtime_probe,
    ).materialize(profile, suite_root)

    # Assert
    assert descriptor["scenario_count"] == 2
    assert (input_folder / "scenario-001.mkv").resolve(strict=True) == first
    assert (input_folder / "scenario-002.mp4").resolve(strict=True) == second
    assert checkpoint_path.is_file()
    assert not pending_path.exists()
    assert (work_root / "full-materialization.json").is_file()


def test_corrupted_partial_context_restarts_only_full_materialization(
    tmp_path: Path,
) -> None:
    """破損したpartial contextだけが再構築されること。

    Arrange:
        - 1本目の確定後に2本目で失敗したfull materializationが用意される
        - Media Runtime context JSONが破損される
    Act:
        - 同じprofileでmaterializationが再開される
    Assert:
        - checkpoint自身がidentityを持つ1本目が再利用されること
        - 未完了の2本目だけが再probeされること
        - suite resetなしで終端descriptorが確定されること
    """
    # Arrange
    profile = _profile(tmp_path)
    profile.input_root.mkdir()
    first = profile.input_root / "private-chapter-01.mkv"
    second = profile.input_root / "private-chapter-02.mp4"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    calls: list[str] = []
    second_failed = False

    def probe(path: Path) -> dict[str, object]:
        nonlocal second_failed
        calls.append(path.name)
        if path == second and not second_failed:
            second_failed = True
            raise OSError("simulated second source failure")
        return {
            "start": Fraction(0),
            "duration": Fraction(50),
            "end": Fraction(50),
        }

    suite_root = profile.artifact_root / "full"
    materializer = FullSuiteMaterializer(
        media_probe=probe,
        media_runtime_probe=_media_runtime,
    )
    with pytest.raises(OSError, match="simulated second source failure"):
        materializer.materialize(profile, suite_root)
    context_path = suite_root / "work" / "full-materialization-context.json"
    context_path.write_text("{broken", encoding="utf-8")

    # Act
    _input_folder, descriptor = materializer.materialize(profile, suite_root)

    # Assert
    assert calls == [
        "private-chapter-01.mkv",
        "private-chapter-02.mp4",
        "private-chapter-02.mp4",
    ]
    assert descriptor["scenario_count"] == 2


def test_failed_runtime_replacement_preserves_previous_source_checkpoints(
    tmp_path: Path,
) -> None:
    """新runtimeでのprobe失敗時に旧source checkpointが保持されること。

    Arrange:
        - 旧Media Runtimeで1本だけ確定した未完了materializationが用意される
        - 新Media Runtimeの最初のsource probeが失敗する
    Act:
        - 新runtimeでfull materializationが再開される
    Assert:
        - 旧symlink targetと全checkpoint bytesが変更されないこと
    """
    # Arrange
    profile = _profile(tmp_path)
    profile.input_root.mkdir()
    first = profile.input_root / "private-chapter-01.mkv"
    second = profile.input_root / "private-chapter-02.mp4"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    suite_root = profile.artifact_root / "full"

    def old_probe(path: Path) -> dict[str, object]:
        if path == second:
            raise OSError("injected old incomplete materialization")
        return {
            "start": Fraction(0),
            "duration": Fraction(50),
            "end": Fraction(50),
        }

    with pytest.raises(OSError, match="injected old incomplete materialization"):
        FullSuiteMaterializer(
            media_probe=old_probe,
            media_runtime_probe=_media_runtime,
        ).materialize(profile, suite_root)
    work_root = suite_root / "work"
    input_path = work_root / "input" / "scenario-001.mkv"
    checkpoint_path = work_root / "source-checkpoints" / ".scenario-001.checkpoint.json"
    old_target = input_path.readlink()
    old_checkpoint = checkpoint_path.read_bytes()

    def fail_probe(_path: Path) -> dict[str, object]:
        raise OSError("injected replacement failure")

    # Act / Assert
    with pytest.raises(OSError, match="injected replacement failure"):
        FullSuiteMaterializer(
            media_probe=fail_probe,
            media_runtime_probe=lambda: MediaRuntimeIdentity(
                "8.0",
                "8.0",
                "c" * 64,
            ),
        ).materialize(profile, suite_root)
    assert input_path.readlink() == old_target
    assert checkpoint_path.read_bytes() == old_checkpoint


def _profile(tmp_path: Path) -> AcceptanceProfile:
    """2本100秒を期待するfull suite profileを返す。"""
    return AcceptanceProfile(
        profile_version="1.0.0",
        input_root=tmp_path / "input",
        configuration_path=tmp_path / "config.toml",
        artifact_root=tmp_path / "artifacts",
        release_expected_total_duration=Fraction(1),
        release_boundary_tolerance_seconds=Fraction(0),
        release_intervals=(
            ReleaseInterval("placeholder.mkv", Fraction(0), Fraction(1), "test"),
        ),
        full_expected_video_count=2,
        full_expected_total_duration=Fraction(100),
        full_duration_tolerance_seconds=Fraction(0),
        profile_digest="b" * 64,
    )


def _media_runtime() -> MediaRuntimeIdentity:
    """test用の固定Media Runtime Identityを返す。"""
    return MediaRuntimeIdentity("7.1", "7.1", "b" * 64)


def _changed_media_runtime() -> MediaRuntimeIdentity:
    """test用の変更後Media Runtime Identityを返す。"""
    return MediaRuntimeIdentity("7.2", "7.2", "d" * 64)
