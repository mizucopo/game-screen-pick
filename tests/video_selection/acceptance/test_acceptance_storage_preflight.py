"""Acceptance storage preflightのtest。"""

from fractions import Fraction
from pathlib import Path

import pytest

from src.video_selection.acceptance.acceptance_profile import AcceptanceProfile
from src.video_selection.acceptance.acceptance_storage_preflight import (
    REQUIRED_ARTIFACT_CAPACITY_BYTES,
    preflight_acceptance_storage,
)
from src.video_selection.acceptance.release_interval import ReleaseInterval


def test_input_size_and_available_capacity_are_recorded(tmp_path: Path) -> None:
    """長時間suite開始前にinput規模とartifact空き容量が記録されること。

    Arrange:
        - 2本のvideoと十分なartifact空き容量を持つprofileが用意される
    Act:
        - storage preflightが実行される
    Assert:
        - video件数、合計byte、利用可能byte、必要容量が返されること
    """
    # Arrange
    profile = _profile(tmp_path, (b"a" * 10, b"b" * 20))
    available = REQUIRED_ARTIFACT_CAPACITY_BYTES + 1

    # Act
    result = preflight_acceptance_storage(
        profile,
        profile.input_root,
        disk_usage_probe=lambda _path: (available * 2, 0, available),
    )

    # Assert
    assert result["input_video_count"] == 2
    assert result["input_video_bytes"] == 30
    assert result["artifact_available_bytes"] == available
    assert (
        result["required_artifact_capacity_bytes"] == REQUIRED_ARTIFACT_CAPACITY_BYTES
    )


def test_insufficient_capacity_is_rejected_before_execution(tmp_path: Path) -> None:
    """cacheとtemporary budgetの合計未満ではacceptanceが開始されないこと。

    Arrange:
        - 必要容量より1 byte少ないartifact filesystemが用意される
    Act:
        - storage preflightが実行される
    Assert:
        - 容量不足として拒否されること
    """
    # Arrange
    profile = _profile(tmp_path, (b"video",))
    available = REQUIRED_ARTIFACT_CAPACITY_BYTES - 1

    # Act / Assert
    with pytest.raises(ValueError, match="容量が不足"):
        preflight_acceptance_storage(
            profile,
            profile.input_root,
            disk_usage_probe=lambda _path: (available * 2, 0, available),
        )


def test_materialized_release_input_is_measured_instead_of_full_root(
    tmp_path: Path,
) -> None:
    """release preflightがfull input rootではなくmaterialize済みclipを測ること。

    Arrange:
        - 2本30 byteのfull rootと1本7 byteのrelease inputが用意される
    Act:
        - release inputを対象にstorage preflightが実行される
    Assert:
        - release clipだけの件数とbyteが記録されること
    """
    # Arrange
    profile = _profile(tmp_path, (b"a" * 10, b"b" * 20))
    release_input = tmp_path / "release-input"
    release_input.mkdir()
    (release_input / "scenario-001.mkv").write_bytes(b"release")
    available = REQUIRED_ARTIFACT_CAPACITY_BYTES + 1

    # Act
    result = preflight_acceptance_storage(
        profile,
        release_input,
        disk_usage_probe=lambda _path: (available * 2, 0, available),
    )

    # Assert
    assert result["input_video_count"] == 1
    assert result["input_video_bytes"] == 7


def _profile(tmp_path: Path, videos: tuple[bytes, ...]) -> AcceptanceProfile:
    """storage test用の最小profileを返す。"""
    input_root = tmp_path / "input"
    input_root.mkdir()
    for index, content in enumerate(videos, start=1):
        (input_root / f"video-{index}.mp4").write_bytes(content)
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    configuration = tmp_path / "configuration.toml"
    configuration.write_text('config_version = "1.0.0"\n', encoding="utf-8")
    return AcceptanceProfile(
        profile_version="1.0.0",
        input_root=input_root,
        configuration_path=configuration,
        artifact_root=artifact_root,
        release_expected_total_duration=Fraction(1),
        release_boundary_tolerance_seconds=Fraction(0),
        release_intervals=(
            ReleaseInterval(
                relative_video_path="video-1.mp4",
                start=Fraction(0),
                end=Fraction(1),
                scenario_role="test",
            ),
        ),
        full_expected_video_count=len(videos),
        full_expected_total_duration=Fraction(1),
        full_duration_tolerance_seconds=Fraction(0),
        profile_digest="a" * 64,
    )
