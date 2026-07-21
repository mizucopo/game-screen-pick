"""release acceptance intervalを匿名stream-copy clipへmaterializeする。"""

import hashlib
import json
import shutil
import subprocess
from collections.abc import Callable, Mapping
from fractions import Fraction
from pathlib import Path
from typing import cast

from .acceptance_profile import AcceptanceProfile
from .atomic_json import read_json_object, write_atomic_json

CommandRunner = Callable[[list[str]], None]
MediaProbe = Callable[[Path], Mapping[str, object]]
ContentDigester = Callable[[Path], str]

_MATERIALIZATION_SCHEMA = "game-screen-pick/release-materialization@2.0.0"


class ReleaseSuiteMaterializer:
    """private intervalを全stream保持の匿名clipへ一度だけ確定する。"""

    def __init__(
        self,
        *,
        command_runner: CommandRunner | None = None,
        media_probe: MediaProbe | None = None,
        content_digester: ContentDigester | None = None,
    ) -> None:
        self._command_runner = command_runner or _run_command
        self._media_probe = media_probe or _probe_media
        self._content_digester = content_digester or _content_digest

    def materialize(
        self,
        profile: AcceptanceProfile,
        suite_root: Path,
    ) -> tuple[Path, dict[str, object]]:
        """clip input folderとpath非依存suite descriptorを返す。"""
        work_root = suite_root / "work"
        input_folder = work_root / "input"
        manifest_path = work_root / "release-materialization.json"
        existing = read_json_object(manifest_path)
        if existing is not None:
            return input_folder, self._restore_existing(
                profile,
                input_folder,
                existing,
            )
        if input_folder.exists():
            raise ValueError("Release suite workが未確定です。--reset-suiteが必要です")
        input_folder.mkdir(parents=True)
        try:
            clips = tuple(
                self._materialize_interval(profile, input_folder, index)
                for index in range(len(profile.release_intervals))
            )
            descriptor = _suite_descriptor(clips)
            manifest = {
                "schema": _MATERIALIZATION_SCHEMA,
                "profile_digest": profile.profile_digest,
                "descriptor": descriptor,
            }
            write_atomic_json(manifest_path, manifest)
            return input_folder, descriptor
        except BaseException:
            shutil.rmtree(input_folder, ignore_errors=True)
            raise

    def _materialize_interval(
        self,
        profile: AcceptanceProfile,
        input_folder: Path,
        zero_based_index: int,
    ) -> dict[str, object]:
        """一つのintervalをstream copyして実測境界を検証する。"""
        interval = profile.release_intervals[zero_based_index]
        source = profile.input_root / interval.relative_video_path
        _require_source_within_root(profile.input_root, source)
        if not source.is_file():
            raise ValueError("Release interval sourceが存在しません")
        source_probe = self._media_probe(source)
        output = input_folder / f"scenario-{zero_based_index + 1:03d}.mkv"
        self._command_runner(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                _ffmpeg_time(interval.start),
                "-i",
                str(source),
                "-to",
                _ffmpeg_time(interval.end),
                "-map",
                "0",
                "-c",
                "copy",
                "-copyts",
                "-avoid_negative_ts",
                "disabled",
                "-map_metadata",
                "-1",
                "-map_chapters",
                "-1",
                "-fflags",
                "+bitexact",
                "-y",
                str(output),
            ]
        )
        clip_probe = self._media_probe(output)
        source_start = _probe_fraction(source_probe, "start")
        clip_start = _probe_fraction(clip_probe, "start")
        clip_duration = _probe_fraction(clip_probe, "duration")
        actual_start = clip_start - source_start
        actual_end = clip_start + clip_duration - source_start
        duration = actual_end - actual_start
        tolerance = profile.release_boundary_tolerance_seconds
        if (
            duration <= 0
            or abs(actual_start - interval.start) > tolerance
            or abs(actual_end - interval.end) > tolerance
            or _probe_streams(source_probe) != _probe_streams(clip_probe)
        ):
            raise ValueError("Release clipの実測境界またはstream保持が不正です")
        return {
            "scenario_id": f"scenario-{zero_based_index + 1:03d}",
            "start": _fraction_record(actual_start),
            "end": _fraction_record(actual_end),
            "duration": _fraction_record(duration),
            "content_sha256": self._content_digester(output),
            "stream_count": len(_probe_streams(clip_probe)),
        }

    def _restore_existing(
        self,
        profile: AcceptanceProfile,
        input_folder: Path,
        manifest: dict[str, object],
    ) -> dict[str, object]:
        """同じprofileの確定済み匿名clipだけをresume用に再利用する。"""
        descriptor = manifest.get("descriptor")
        if (
            manifest.get("schema") != _MATERIALIZATION_SCHEMA
            or manifest.get("profile_digest") != profile.profile_digest
            or not isinstance(descriptor, dict)
            or not input_folder.is_dir()
        ):
            raise ValueError("Release suite stateがprofileと一致しません")
        clips = descriptor.get("clips")
        if not isinstance(clips, list) or len(clips) != len(profile.release_intervals):
            raise ValueError("Release suite clip manifestが不正です")
        for index, item in enumerate(clips, start=1):
            if not isinstance(item, dict):
                raise ValueError("Release suite clip manifestが不正です")
            path = input_folder / f"scenario-{index:03d}.mkv"
            if not path.is_file() or item.get(
                "content_sha256"
            ) != self._content_digester(path):
                raise ValueError("Release suite clipが変更されています")
        return cast(dict[str, object], descriptor)


def _suite_descriptor(clips: tuple[dict[str, object], ...]) -> dict[str, object]:
    total = sum(
        (_record_fraction(item["duration"]) for item in clips),
        start=Fraction(0),
    )
    canonical = json.dumps(clips, sort_keys=True, separators=(",", ":")).encode()
    return {
        "suite_fingerprint": hashlib.sha256(canonical).hexdigest(),
        "scenario_count": len(clips),
        "total_duration": _fraction_record(total),
        "clips": list(clips),
    }


def _require_source_within_root(root: Path, source: Path) -> None:
    try:
        source.resolve(strict=False).relative_to(root.resolve(strict=True))
    except (OSError, ValueError):
        msg = "Release interval sourceがinput root外を指しています"
        raise ValueError(msg) from None


def _run_command(command: list[str]) -> None:
    try:
        subprocess.run(command, check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        raise ValueError("Release clipのFFmpeg stream copyに失敗しました") from None


def _probe_media(path: Path) -> Mapping[str, object]:
    """startとcopyts Matroskaのend timestampを含むmedia probeを返す。"""
    try:
        process = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=start_time,duration:stream=codec_type,codec_name",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        value: object = json.loads(process.stdout)
    except (OSError, subprocess.CalledProcessError, TypeError, ValueError):
        raise ValueError("Release mediaをffprobeできません") from None
    if not isinstance(value, dict):
        raise ValueError("FFprobe resultがobjectではありません")
    format_value = value.get("format")
    streams_value = value.get("streams")
    if not isinstance(format_value, dict) or not isinstance(streams_value, list):
        raise ValueError("FFprobe resultにformatまたはstreamがありません")
    streams: list[tuple[str, str]] = []
    for item in streams_value:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("codec_type"), str)
            or not isinstance(item.get("codec_name"), str)
        ):
            raise ValueError("FFprobe streamが不正です")
        streams.append((item["codec_type"], item["codec_name"]))
    return {
        "start": Fraction(str(format_value.get("start_time", "0"))),
        "duration": Fraction(str(format_value["duration"])),
        "streams": tuple(sorted(streams)),
    }


def _probe_fraction(value: Mapping[str, object], key: str) -> Fraction:
    result = value.get(key)
    if not isinstance(result, Fraction):
        raise ValueError("Media probeの時刻が不正です")
    return result


def _probe_streams(value: Mapping[str, object]) -> tuple[tuple[str, str], ...]:
    streams = value.get("streams")
    if not isinstance(streams, tuple) or any(
        not isinstance(item, tuple)
        or len(item) != 2
        or any(not isinstance(part, str) for part in item)
        for item in streams
    ):
        raise ValueError("Media probeのstreamが不正です")
    return cast(tuple[tuple[str, str], ...], streams)


def _content_digest(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _ffmpeg_time(value: Fraction) -> str:
    return f"{float(value):.6f}"


def _fraction_record(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _record_fraction(value: object) -> Fraction:
    if not isinstance(value, dict):
        raise ValueError("Duration recordが不正です")
    numerator = value.get("numerator")
    denominator = value.get("denominator")
    if not isinstance(numerator, int) or not isinstance(denominator, int):
        raise ValueError("Duration recordが不正です")
    return Fraction(numerator, denominator)
