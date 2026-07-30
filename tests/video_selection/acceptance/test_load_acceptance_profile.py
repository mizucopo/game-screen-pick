"""target-only acceptance profile loaderのtest。"""

from fractions import Fraction
from pathlib import Path

import pytest

from src.video_selection.acceptance.load_acceptance_profile import (
    load_acceptance_profile,
)


def test_strict_profile_resolves_paths_suites_intervals_and_digest(
    tmp_path: Path,
) -> None:
    """strict profileがpath、suite期待値、interval、digestへ変換されること。

    Arrange:
        - 3つのabsolute pathとrelease/full suiteを持つTOMLが用意される
    Act:
        - target-only acceptance profileが読み込まれる
    Assert:
        - path、ISO duration、tolerance、interval、digestが保持されること
    """
    # Arrange
    path = tmp_path / "target.toml"
    path.write_text(_profile_text(tmp_path), encoding="utf-8")

    # Act
    profile = load_acceptance_profile(path)

    # Assert
    assert profile.input_root == tmp_path / "input"
    assert profile.configuration_path == tmp_path / "config.toml"
    assert profile.artifact_root == tmp_path / "artifacts"
    assert profile.release_expected_total_duration == Fraction(1800)
    assert profile.release_boundary_tolerance_seconds == Fraction(5)
    assert profile.release_intervals[0].expected_duration == Fraction(1800)
    assert profile.full_expected_video_count == 12
    assert profile.full_expected_total_duration == Fraction(182400)
    assert len(profile.profile_digest) == 64


def test_unknown_profile_key_is_rejected(tmp_path: Path) -> None:
    """target profileの未知keyが無視されず拒否されること。

    Arrange:
        - rootへ未知のprivate settingを追加したTOMLが用意される
    Act:
        - acceptance profileの読み込みが試行される
    Assert:
        - strict schema違反としてValueErrorになること
    """
    # Arrange
    path = tmp_path / "target.toml"
    path.write_text(
        _profile_text(tmp_path) + "\nsecret_mode = true\n",
        encoding="utf-8",
    )

    # Act
    # Assert
    with pytest.raises(ValueError, match="strict schema"):
        load_acceptance_profile(path)


def test_release_interval_cannot_escape_input_root(tmp_path: Path) -> None:
    """parent segmentを持つrelease source pathが拒否されること。

    Arrange:
        - input root外を指すrelative_video_pathが用意される
    Act:
        - acceptance profileの読み込みが試行される
    Assert:
        - Release intervalのpath contract違反になること
    """
    # Arrange
    path = tmp_path / "target.toml"
    path.write_text(
        _profile_text(tmp_path).replace("chapter.mkv", "../chapter.mkv"),
        encoding="utf-8",
    )

    # Act
    # Assert
    with pytest.raises(ValueError, match="Release interval"):
        load_acceptance_profile(path)


def test_committed_profile_template_matches_strict_schema(tmp_path: Path) -> None:
    """公開templateが実値置換後にstrict profileとして読み込まれること。

    Arrange:
        - committed templateのplaceholderをprivate相当のtest pathへ置換した
          copyが用意される
    Act:
        - target acceptance profile loaderでcopyが読み込まれる
    Assert:
        - release 30分、full 12本50時間40分のprofileになること
    """
    # Arrange
    template = Path("docs/examples/target-acceptance.toml").read_text(
        encoding="utf-8",
    )
    rendered = (
        template.replace("<absolute-path-on-supported-target>", str(tmp_path / "input"))
        .replace(
            "<absolute-path-to-video-selection-toml>",
            str(tmp_path / "video-selection.toml"),
        )
        .replace("<absolute-private-artifact-root>", str(tmp_path / "artifacts"))
        .replace("<relative-video-path>", "source.mkv")
    )
    profile_path = tmp_path / "target.toml"
    profile_path.write_text(rendered, encoding="utf-8")

    # Act
    profile = load_acceptance_profile(profile_path)

    # Assert
    assert profile.release_expected_total_duration == Fraction(1800)
    assert len(profile.release_intervals) == 3
    assert profile.full_expected_video_count == 12
    assert profile.full_expected_total_duration == Fraction(182400)


def _profile_text(tmp_path: Path) -> str:
    """validなtarget-only profile TOMLを返す。"""
    return f'''profile_version = "1.0.0"
input_root = "{tmp_path / "input"}"
configuration_path = "{tmp_path / "config.toml"}"
artifact_root = "{tmp_path / "artifacts"}"

[release_suite]
expected_total_duration = "PT30M"
boundary_tolerance_seconds = 5

[[release_suite.intervals]]
relative_video_path = "chapter.mkv"
start = "PT0S"
end = "PT30M"
scenario_role = "representative-gameplay"

[full_scale_suite]
expected_video_count = 12
expected_total_duration = "PT50H40M"
duration_tolerance_seconds = 60
'''
