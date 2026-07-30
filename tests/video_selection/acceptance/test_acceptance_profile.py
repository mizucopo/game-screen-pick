"""AcceptanceProfile modelのtest。"""

from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from src.video_selection.acceptance.acceptance_profile import AcceptanceProfile
from src.video_selection.acceptance.release_interval import ReleaseInterval


def test_release_interval_sum_must_match_expected_duration_within_tolerance(
    tmp_path: Path,
) -> None:
    """release interval合計がprofile期待値のtolerance内だけで受理されること。

    Arrange:
        - 60秒intervalと60秒期待値を持つprofileが用意される
    Act:
        - 期待値を70秒へ変更したprofileの構築が試行される
    Assert:
        - tolerance 1秒を超える不一致として拒否されること
    """
    # Arrange
    profile = _profile(tmp_path)

    # Act
    with pytest.raises(ValueError) as error:
        replace(profile, release_expected_total_duration=Fraction(70))

    # Assert
    assert "Acceptance profile" in str(error.value)


@pytest.mark.parametrize("artifact_suffix", (Path(), Path("artifacts")))
def test_artifact_root_cannot_equal_or_descend_from_input_root(
    tmp_path: Path,
    artifact_suffix: Path,
) -> None:
    """artifact rootがinput root自身または配下に置かれないこと。

    Arrange:
        - valid profileとinput root自身または子folderのartifact pathが用意される
    Act:
        - artifact rootを重ねたprofileの構築が試行される
    Assert:
        - recursive full source discoveryへ生成物が混入する構成として拒否されること
    """
    # Arrange
    profile = _profile(tmp_path)
    artifact_root = profile.input_root / artifact_suffix

    # Act
    with pytest.raises(ValueError) as error:
        replace(profile, artifact_root=artifact_root)

    # Assert
    assert "Acceptance profile" in str(error.value)


def _profile(tmp_path: Path) -> AcceptanceProfile:
    """validな最小profileを返す。"""
    return AcceptanceProfile(
        profile_version="1.0.0",
        input_root=tmp_path / "input",
        configuration_path=tmp_path / "config.toml",
        artifact_root=tmp_path / "artifacts",
        release_expected_total_duration=Fraction(60),
        release_boundary_tolerance_seconds=Fraction(1),
        release_intervals=(
            ReleaseInterval("video.mkv", Fraction(0), Fraction(60), "event"),
        ),
        full_expected_video_count=1,
        full_expected_total_duration=Fraction(60),
        full_duration_tolerance_seconds=Fraction(1),
        profile_digest="a" * 64,
    )
