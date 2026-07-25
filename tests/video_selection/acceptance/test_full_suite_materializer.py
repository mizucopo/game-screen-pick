"""full-scale suite匿名input materializerのtest。"""

from fractions import Fraction
from pathlib import Path

import pytest

from src.video_selection.acceptance.acceptance_profile import AcceptanceProfile
from src.video_selection.acceptance.full_suite_materializer import (
    FullSuiteMaterializer,
)
from src.video_selection.acceptance.release_interval import ReleaseInterval


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
        media_probe=lambda _path: {
            "start": Fraction(0),
            "duration": Fraction(50),
            "streams": (("video", "h264"),),
        }
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
        media_probe=lambda _path: {
            "start": Fraction(0),
            "duration": Fraction(50),
        }
    )
    suite_root = profile.artifact_root / "full"
    materializer.materialize(profile, suite_root)
    first.write_bytes(b"changed")

    # Act / Assert
    with pytest.raises(ValueError, match="source"):
        materializer.materialize(profile, suite_root)


def test_nonzero_media_start_is_subtracted_from_full_duration(
    tmp_path: Path,
) -> None:
    """非0 media startがfull suiteの経過時間から除かれること。

    Arrange:
        - start 5秒、end timestamp 55秒の2動画が用意される
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
        media_probe=lambda _path: {
            "start": Fraction(5),
            "duration": Fraction(55),
        }
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
        media_probe=lambda _path: {
            "start": Fraction(0),
            "duration": Fraction(50),
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
        media_probe=lambda _path: {
            "start": Fraction(0),
            "duration": Fraction(50),
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


def test_duration_mismatch_removes_partial_anonymous_view(tmp_path: Path) -> None:
    """full duration preflight failureでpartial symlink viewが削除されること。

    Arrange:
        - profile期待値と異なるdurationを返す2本のsourceが用意される
    Act:
        - full suite materializationが試行される
    Assert:
        - preflight failureとなり匿名input folderが残らないこと
    """
    # Arrange
    profile = _profile(tmp_path)
    profile.input_root.mkdir()
    (profile.input_root / "private-chapter-01.mkv").write_bytes(b"first")
    (profile.input_root / "private-chapter-02.mp4").write_bytes(b"second")
    suite_root = profile.artifact_root / "full"
    materializer = FullSuiteMaterializer(
        media_probe=lambda _path: {
            "start": Fraction(0),
            "duration": Fraction(40),
        },
    )

    # Act / Assert
    with pytest.raises(ValueError, match="duration"):
        materializer.materialize(profile, suite_root)
    assert not (suite_root / "work" / "input").exists()


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
