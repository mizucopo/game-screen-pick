"""target-only acceptance TOMLのstrict loader。"""

import hashlib
import re
import tomllib
from fractions import Fraction
from pathlib import Path
from typing import cast

from .acceptance_profile import AcceptanceProfile
from .release_interval import ReleaseInterval

_DURATION_PATTERN = re.compile(
    r"PT(?:(?P<hours>[0-9]+)H)?(?:(?P<minutes>[0-9]+)M)?"
    r"(?:(?P<seconds>[0-9]+(?:\.[0-9]+)?)S)?"
)


def load_acceptance_profile(path: Path) -> AcceptanceProfile:
    """未知keyを拒否しprivate profileを検証済みmodelへ変換する。"""
    try:
        raw_bytes = path.read_bytes()
        value = cast(dict[str, object], tomllib.loads(raw_bytes.decode("utf-8")))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError):
        raise ValueError("Acceptance profileを読み込めません") from None
    _require_keys(
        value,
        required={
            "profile_version",
            "input_root",
            "configuration_path",
            "artifact_root",
            "release_suite",
            "full_scale_suite",
        },
        location="root",
    )
    release = _table(value["release_suite"], "release_suite")
    _require_keys(
        release,
        required={
            "expected_total_duration",
            "boundary_tolerance_seconds",
            "intervals",
        },
        location="release_suite",
    )
    full = _table(value["full_scale_suite"], "full_scale_suite")
    _require_keys(
        full,
        required={
            "expected_video_count",
            "expected_total_duration",
            "duration_tolerance_seconds",
        },
        location="full_scale_suite",
    )
    intervals_value = release["intervals"]
    if not isinstance(intervals_value, list):
        raise ValueError("release_suite.intervalsにはtable arrayが必要です")
    intervals = tuple(
        _release_interval(item, index)
        for index, item in enumerate(intervals_value, start=1)
    )
    return AcceptanceProfile(
        profile_version=_string(value["profile_version"], "profile_version"),
        input_root=Path(_string(value["input_root"], "input_root")),
        configuration_path=Path(
            _string(value["configuration_path"], "configuration_path")
        ),
        artifact_root=Path(_string(value["artifact_root"], "artifact_root")),
        release_expected_total_duration=_duration(
            release["expected_total_duration"],
            "release_suite.expected_total_duration",
        ),
        release_boundary_tolerance_seconds=_non_negative_fraction(
            release["boundary_tolerance_seconds"],
            "release_suite.boundary_tolerance_seconds",
        ),
        release_intervals=intervals,
        full_expected_video_count=_positive_integer(
            full["expected_video_count"],
            "full_scale_suite.expected_video_count",
        ),
        full_expected_total_duration=_duration(
            full["expected_total_duration"],
            "full_scale_suite.expected_total_duration",
        ),
        full_duration_tolerance_seconds=_non_negative_fraction(
            full["duration_tolerance_seconds"],
            "full_scale_suite.duration_tolerance_seconds",
        ),
        profile_digest=hashlib.sha256(raw_bytes).hexdigest(),
    )


def _release_interval(value: object, index: int) -> ReleaseInterval:
    item = _table(value, f"release_suite.intervals[{index}]")
    _require_keys(
        item,
        required={"relative_video_path", "start", "end", "scenario_role"},
        location=f"release_suite.intervals[{index}]",
    )
    return ReleaseInterval(
        relative_video_path=_string(
            item["relative_video_path"],
            f"release_suite.intervals[{index}].relative_video_path",
        ),
        start=_duration(item["start"], f"release_suite.intervals[{index}].start"),
        end=_duration(item["end"], f"release_suite.intervals[{index}].end"),
        scenario_role=_string(
            item["scenario_role"],
            f"release_suite.intervals[{index}].scenario_role",
        ),
    )


def _duration(value: object, location: str) -> Fraction:
    text = _string(value, location)
    match = _DURATION_PATTERN.fullmatch(text)
    if match is None or not any(match.groupdict().values()):
        raise ValueError(f"{location}にはISO 8601 durationが必要です")
    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes") or 0)
    seconds = Fraction(match.group("seconds") or "0")
    result = Fraction(hours * 3600 + minutes * 60) + seconds
    if result < 0:
        raise ValueError(f"{location}には非負durationが必要です")
    return result


def _non_negative_fraction(value: object, location: str) -> Fraction:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{location}にはnumberが必要です")
    result = Fraction(str(value))
    if result < 0:
        raise ValueError(f"{location}には0以上が必要です")
    return result


def _positive_integer(value: object, location: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{location}には正の整数が必要です")
    return value


def _string(value: object, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}には空でないstringが必要です")
    return value


def _table(value: object, location: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{location}にはtableが必要です")
    return cast(dict[str, object], value)


def _require_keys(
    value: dict[str, object],
    *,
    required: set[str],
    location: str,
) -> None:
    actual = set(value)
    if actual != required:
        raise ValueError(f"{location}のkeyがstrict schemaと一致しません")
