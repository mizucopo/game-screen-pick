"""release suite materializerのFFmpeg integration test。"""

from fractions import Fraction
from pathlib import Path

from src.video_selection.acceptance.acceptance_profile import AcceptanceProfile
from src.video_selection.acceptance.release_interval import ReleaseInterval
from src.video_selection.acceptance.release_suite_materializer import (
    ReleaseSuiteMaterializer,
)
from tests_ffmpeg.support.ffmpeg_fixture_factory import generate_nonzero_start_video


def test_nonzero_source_start_preserves_relative_clip_end(tmp_path: Path) -> None:
    """非0 source startでも相対終了境界までclipが作成されること。

    Arrange:
        - 5秒の非0開始PTSを持つ4秒videoと1〜3秒のintervalが用意される
    Act:
        - 実FFmpegでrelease suiteがmaterializeされる
    Assert:
        - clipの相対終了境界が3秒として記録されること
    """
    # Arrange
    input_root = tmp_path / "input"
    input_root.mkdir()
    source = generate_nonzero_start_video(input_root / "source.mkv")
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
