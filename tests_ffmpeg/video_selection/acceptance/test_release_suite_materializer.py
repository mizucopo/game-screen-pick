"""release suite materializerのFFmpeg integration test。"""

from fractions import Fraction
from pathlib import Path

import pytest

from src.video_selection.acceptance.acceptance_profile import AcceptanceProfile
from src.video_selection.acceptance.full_suite_materializer import (
    FullSuiteMaterializer,
)
from src.video_selection.acceptance.release_interval import ReleaseInterval
from src.video_selection.acceptance.release_suite_materializer import (
    ReleaseSuiteMaterializer,
)
from tests_ffmpeg.support.ffmpeg_fixture_factory import generate_nonzero_start_video


@pytest.mark.parametrize("suffix", (".mkv", ".mp4"))
def test_nonzero_source_start_preserves_relative_clip_end(
    tmp_path: Path,
    suffix: str,
) -> None:
    """container差があっても非0 source startの相対終了境界が保持されること。

    Arrange:
        - 5秒の非0開始PTSを持つMatroskaまたはMP4と1〜3秒のintervalが用意される
    Act:
        - 実FFmpegでrelease suiteがmaterializeされる
    Assert:
        - clipの相対終了境界が3秒として記録されること
    """
    # Arrange
    input_root = tmp_path / "input"
    input_root.mkdir()
    source = generate_nonzero_start_video(input_root / f"source{suffix}")
    profile = AcceptanceProfile(
        profile_version="1.0.0",
        input_root=input_root,
        configuration_path=tmp_path / "config.toml",
        artifact_root=tmp_path / "artifacts",
        release_expected_total_duration=Fraction(2),
        release_boundary_tolerance_seconds=Fraction(1),
        release_intervals=(
            ReleaseInterval(source.name, Fraction(1), Fraction(3), "test"),
        ),
        full_expected_video_count=1,
        full_expected_total_duration=Fraction(4),
        full_duration_tolerance_seconds=Fraction(0),
        profile_digest="c" * 64,
    )

    # Act
    _, descriptor = ReleaseSuiteMaterializer().materialize(
        profile,
        profile.artifact_root / "release",
    )

    # Assert
    clips = descriptor["clips"]
    assert isinstance(clips, list)
    clip = clips[0]
    assert isinstance(clip, dict)
    assert clip["end"] == {"numerator": 3, "denominator": 1}


def test_full_suite_normalizes_nonzero_start_duration_across_containers(
    tmp_path: Path,
) -> None:
    """MatroskaとMP4の非0開始時刻が同じ経過時間へ正規化されること。

    Arrange:
        - 5秒開始で経過4秒のMatroskaとMP4が用意される
    Act:
        - 実FFprobeでfull suiteがmaterializeされる
    Assert:
        - containerごとのduration表現差を除いた合計8秒が記録されること
    """
    # Arrange
    input_root = tmp_path / "input"
    input_root.mkdir()
    mkv_source = generate_nonzero_start_video(input_root / "source.mkv")
    generate_nonzero_start_video(input_root / "source.mp4")
    profile = AcceptanceProfile(
        profile_version="1.0.0",
        input_root=input_root,
        configuration_path=tmp_path / "config.toml",
        artifact_root=tmp_path / "artifacts",
        release_expected_total_duration=Fraction(1),
        release_boundary_tolerance_seconds=Fraction(1),
        release_intervals=(
            ReleaseInterval(mkv_source.name, Fraction(0), Fraction(1), "test"),
        ),
        full_expected_video_count=2,
        full_expected_total_duration=Fraction(8),
        full_duration_tolerance_seconds=Fraction(0),
        profile_digest="d" * 64,
    )

    # Act
    _, descriptor = FullSuiteMaterializer().materialize(
        profile,
        profile.artifact_root / "full",
    )

    # Assert
    assert descriptor["total_duration"] == {"numerator": 8, "denominator": 1}
