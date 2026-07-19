"""release suite stream-copy materializerのtest。"""

from fractions import Fraction
from pathlib import Path

import pytest

from src.video_selection.acceptance.acceptance_profile import AcceptanceProfile
from src.video_selection.acceptance.release_interval import ReleaseInterval
from src.video_selection.acceptance.release_suite_materializer import (
    ReleaseSuiteMaterializer,
)


def test_anonymous_clips_preserve_all_streams_and_record_actual_boundaries(
    tmp_path: Path,
) -> None:
    """匿名clipが全streamをcopyし実測境界からsuite identityを作ること。

    Arrange:
        - 2 streamのsource、keyframe差を含むprobe、fake FFmpegが用意される
    Act:
        - release suiteがmaterializeされる
    Assert:
        - map 0/copytsのstream-copy commandが匿名filenameへ実行されること
        - 実測境界、duration、contentからdescriptorが生成されること
        - manifestへsource pathや動画名が記録されないこと
    """
    # Arrange
    profile = _profile(tmp_path)
    profile.input_root.mkdir()
    source = profile.input_root / "private-video.mkv"
    source.write_bytes(b"source")
    commands: list[list[str]] = []

    def run(command: list[str]) -> None:
        commands.append(command)
        Path(command[-1]).write_bytes(b"anonymous-clip")

    def probe(path: Path) -> dict[str, object]:
        return {
            "start": Fraction(0 if path == source else 8),
            "duration": Fraction(100 if path == source else 1809),
            "streams": (("audio", "aac"), ("video", "h264")),
        }

    materializer = ReleaseSuiteMaterializer(
        command_runner=run,
        media_probe=probe,
    )

    # Act
    input_folder, descriptor = materializer.materialize(
        profile,
        profile.artifact_root / "release",
    )

    # Assert
    assert Path(commands[0][-1]).name == "scenario-001.mkv"
    assert commands[0][commands[0].index("-map") + 1] == "0"
    assert commands[0][commands[0].index("-c") + 1] == "copy"
    assert "-copyts" in commands[0]
    assert commands[0][commands[0].index("-map_metadata") + 1] == "-1"
    assert commands[0][commands[0].index("-map_chapters") + 1] == "-1"
    assert commands[0][commands[0].index("-fflags") + 1] == "+bitexact"
    assert commands[0][commands[0].index("-to") + 1] == "1810.000000"
    assert descriptor["scenario_count"] == 1
    assert descriptor["total_duration"] == {"numerator": 1801, "denominator": 1}
    manifest = (input_folder.parent / "release-materialization.json").read_text(
        encoding="utf-8"
    )
    assert "private-video" not in manifest
    assert str(profile.input_root) not in manifest


def test_boundary_outside_tolerance_removes_partial_clips(tmp_path: Path) -> None:
    """実測境界がtoleranceを超えるとpartial clipが残らないこと。

    Arrange:
        - 期待startから10秒ずれたclip probeが用意される
    Act:
        - release suiteのmaterializeが試行される
    Assert:
        - preflight failureとなりinput work folderが削除されること
    """
    # Arrange
    profile = _profile(tmp_path)
    profile.input_root.mkdir()
    source = profile.input_root / "private-video.mkv"
    source.write_bytes(b"source")

    def run(command: list[str]) -> None:
        Path(command[-1]).write_bytes(b"clip")

    def probe(path: Path) -> dict[str, object]:
        return {
            "start": Fraction(0 if path == source else 20),
            "duration": Fraction(100 if path == source else 1820),
            "streams": (("video", "h264"),),
        }

    suite_root = profile.artifact_root / "release"

    # Act / Assert
    with pytest.raises(ValueError, match="実測境界"):
        ReleaseSuiteMaterializer(
            command_runner=run,
            media_probe=probe,
        ).materialize(profile, suite_root)
    assert not (suite_root / "work" / "input").exists()


def test_completed_materialization_is_reused_without_ffmpeg(tmp_path: Path) -> None:
    """同じprofileとcontentの確定済みclipがresume時に再利用されること。

    Arrange:
        - 一度materialize済みのrelease suiteが用意される
    Act:
        - 同じprofileでmaterializeが再度呼ばれる
    Assert:
        - FFmpegを再実行せず同じdescriptorが返されること
    """
    # Arrange
    profile = _profile(tmp_path)
    profile.input_root.mkdir()
    source = profile.input_root / "private-video.mkv"
    source.write_bytes(b"source")
    call_count = 0

    def run(command: list[str]) -> None:
        nonlocal call_count
        call_count += 1
        Path(command[-1]).write_bytes(b"clip")

    def probe(path: Path) -> dict[str, object]:
        return {
            "start": Fraction(0 if path == source else 8),
            "duration": Fraction(100 if path == source else 1809),
            "streams": (("video", "h264"),),
        }

    materializer = ReleaseSuiteMaterializer(
        command_runner=run,
        media_probe=probe,
    )
    suite_root = profile.artifact_root / "release"
    _, first = materializer.materialize(profile, suite_root)

    # Act
    _, second = materializer.materialize(profile, suite_root)

    # Assert
    assert call_count == 1
    assert second == first


def _profile(tmp_path: Path) -> AcceptanceProfile:
    """一つの30分intervalを持つprofileを返す。"""
    return AcceptanceProfile(
        profile_version="1.0.0",
        input_root=tmp_path / "input",
        configuration_path=tmp_path / "config.toml",
        artifact_root=tmp_path / "artifacts",
        release_expected_total_duration=Fraction(1800),
        release_boundary_tolerance_seconds=Fraction(5),
        release_intervals=(
            ReleaseInterval(
                relative_video_path="private-video.mkv",
                start=Fraction(10),
                end=Fraction(1810),
                scenario_role="representative-gameplay",
            ),
        ),
        full_expected_video_count=12,
        full_expected_total_duration=Fraction(182400),
        full_duration_tolerance_seconds=Fraction(60),
        profile_digest="a" * 64,
    )
