"""target-only acceptance profileの検証済みdomain model。"""

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from .release_interval import ReleaseInterval


@dataclass(frozen=True)
class AcceptanceProfile:
    """実pathとsuite期待値をrepository外だけで保持する。"""

    profile_version: str
    input_root: Path
    configuration_path: Path
    artifact_root: Path
    release_expected_total_duration: Fraction
    release_boundary_tolerance_seconds: Fraction
    release_intervals: tuple[ReleaseInterval, ...]
    full_expected_video_count: int
    full_expected_total_duration: Fraction
    full_duration_tolerance_seconds: Fraction
    profile_digest: str

    def __post_init__(self) -> None:
        """version、suite件数、duration、digestの基本契約を検証する。"""
        expected_release_duration = sum(
            (item.expected_duration for item in self.release_intervals),
            start=Fraction(0),
        )
        if (
            self.profile_version != "1.0.0"
            or not self.input_root.is_absolute()
            or not self.configuration_path.is_absolute()
            or not self.artifact_root.is_absolute()
            or _artifact_root_is_within_input(
                self.artifact_root,
                self.input_root,
            )
            or not self.release_intervals
            or self.release_expected_total_duration <= 0
            or self.release_boundary_tolerance_seconds < 0
            or abs(expected_release_duration - self.release_expected_total_duration)
            > self.release_boundary_tolerance_seconds
            or self.full_expected_video_count < 1
            or self.full_expected_total_duration <= 0
            or self.full_duration_tolerance_seconds < 0
            or len(self.profile_digest) != 64
            or any(
                character not in "0123456789abcdef" for character in self.profile_digest
            )
        ):
            msg = "Acceptance profileのversion、path、suite期待値が不正です"
            raise ValueError(msg)


def _artifact_root_is_within_input(artifact_root: Path, input_root: Path) -> bool:
    try:
        artifact_root.resolve(strict=False).relative_to(
            input_root.resolve(strict=False)
        )
    except ValueError:
        return False
    except (OSError, RuntimeError):
        return True
    return True
