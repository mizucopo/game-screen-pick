"""release acceptance intervalを匿名stream-copy clipへmaterializeする。"""

import hashlib
import json
import os
import shutil
import stat
import subprocess
from collections.abc import Callable, Mapping
from fractions import Fraction
from pathlib import Path
from typing import cast
from uuid import uuid4

from ..services.discover_video_paths import discover_video_paths
from .acceptance_profile import AcceptanceProfile
from .atomic_json import read_json_object, write_atomic_json
from .materialization_media_runtime import (
    MediaRuntimeProbe,
    media_runtime_identity_record,
    parse_media_runtime_identity_record,
    probe_media_runtime_identity,
)
from .release_interval import ReleaseInterval
from .source_snapshot_fingerprint import acceptance_source_snapshot_fingerprint
from .suite_owned_directory import validate_suite_owned_directory_chain

CommandRunner = Callable[[list[str]], None]
MediaProbe = Callable[[Path], Mapping[str, object]]
ContentDigester = Callable[[Path], str]

_MATERIALIZATION_SCHEMA = "game-screen-pick/release-materialization@3.0.0"
_INTERVAL_SCHEMA = "game-screen-pick/release-interval-checkpoint@2.0.0"
_CONTEXT_SCHEMA = "game-screen-pick/release-materialization-context@2.0.0"


class ReleaseSuiteMaterializer:
    """private intervalを全stream保持の匿名clipへ一度だけ確定する。"""

    def __init__(
        self,
        *,
        command_runner: CommandRunner | None = None,
        media_probe: MediaProbe | None = None,
        content_digester: ContentDigester | None = None,
        media_runtime_probe: MediaRuntimeProbe = probe_media_runtime_identity,
    ) -> None:
        self._command_runner = command_runner or _run_command
        self._media_probe = media_probe or _probe_media
        self._content_digester = content_digester or _content_digest
        self._media_runtime_probe = media_runtime_probe

    def materialize(
        self,
        profile: AcceptanceProfile,
        suite_root: Path,
    ) -> tuple[Path, dict[str, object]]:
        """clip input folderとpath非依存suite descriptorを返す。"""
        work_root = suite_root / "work"
        input_folder = work_root / "input"
        checkpoint_root = work_root / "interval-checkpoints"
        manifest_path = work_root / "release-materialization.json"
        context_path = work_root / "release-materialization-context.json"
        validate_suite_owned_directory_chain(
            suite_root,
            work_root,
            input_folder,
            suite_label="Release",
        )
        source_snapshot = acceptance_source_snapshot_fingerprint(profile, "release")
        existing = _read_materialization_manifest(manifest_path)
        if existing is not None and _manifest_descriptor_is_valid(profile, existing):
            return input_folder, self._restore_existing(
                profile,
                input_folder,
                existing,
                source_snapshot,
            )
        recovered = self._restore_completed_from_checkpoints(
            profile,
            input_folder,
            checkpoint_root,
            source_snapshot,
        )
        if recovered is not None:
            recovered_runtime, descriptor = recovered
            write_atomic_json(
                manifest_path,
                {
                    "schema": _MATERIALIZATION_SCHEMA,
                    "profile_digest": profile.profile_digest,
                    "media_runtime_identity": recovered_runtime,
                    "descriptor": descriptor,
                },
            )
            return input_folder, descriptor
        media_runtime = media_runtime_identity_record(self._media_runtime_probe())
        _prepare_materialization_context(
            profile,
            input_folder,
            checkpoint_root,
            context_path,
            media_runtime,
        )
        if input_folder.is_symlink() or (
            input_folder.exists() and not input_folder.is_dir()
        ):
            raise ValueError("Release suite workが不正です。--reset-suiteが必要です")
        input_folder.mkdir(parents=True, exist_ok=True)
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        clips = tuple(
            self._materialize_interval(
                profile,
                input_folder,
                checkpoint_root,
                index,
                media_runtime,
            )
            for index in range(len(profile.release_intervals))
        )
        descriptor = _suite_descriptor(clips, source_snapshot)
        if (
            acceptance_source_snapshot_fingerprint(profile, "release")
            != source_snapshot
        ):
            raise ValueError("Release suite sourceがmaterialize中に変更されました")
        _validate_total_duration(profile, descriptor)
        _validate_anonymous_video_names(
            input_folder,
            len(profile.release_intervals),
        )
        manifest = {
            "schema": _MATERIALIZATION_SCHEMA,
            "profile_digest": profile.profile_digest,
            "media_runtime_identity": media_runtime,
            "descriptor": descriptor,
        }
        write_atomic_json(manifest_path, manifest)
        return input_folder, descriptor

    def _restore_completed_from_checkpoints(
        self,
        profile: AcceptanceProfile,
        input_folder: Path,
        checkpoint_root: Path,
        source_snapshot: str,
    ) -> tuple[dict[str, str], dict[str, object]] | None:
        """全intervalが同じ記録済みruntimeで確定済みなら終端manifestを復元する。"""
        if (
            input_folder.is_symlink()
            or checkpoint_root.is_symlink()
            or not input_folder.is_dir()
            or not checkpoint_root.is_dir()
        ):
            return None
        recorded_runtime: dict[str, str] | None = None
        clips: list[dict[str, object]] = []
        for index, interval in enumerate(profile.release_intervals):
            source = profile.input_root / interval.relative_video_path
            _require_source_within_root(profile.input_root, source)
            if not source.is_file():
                return None
            checkpoint_path = (
                checkpoint_root / f".scenario-{index + 1:03d}.checkpoint.json"
            )
            pending_checkpoint_path = (
                checkpoint_root / f".scenario-{index + 1:03d}.pending.json"
            )
            restored_checkpoint = self._restore_recorded_interval(
                profile,
                index,
                _source_snapshot(source),
                input_folder / f"scenario-{index + 1:03d}.mkv",
                checkpoint_path,
                pending_checkpoint_path,
            )
            if restored_checkpoint is None:
                return None
            runtime, restored = restored_checkpoint
            if recorded_runtime is not None and runtime != recorded_runtime:
                return None
            recorded_runtime = runtime
            clips.append(restored)
        if recorded_runtime is None:
            return None
        descriptor = _suite_descriptor(tuple(clips), source_snapshot)
        if (
            acceptance_source_snapshot_fingerprint(profile, "release")
            != source_snapshot
        ):
            raise ValueError("Release suite sourceがmaterialize中に変更されました")
        _validate_total_duration(profile, descriptor)
        _validate_anonymous_video_names(
            input_folder,
            len(profile.release_intervals),
        )
        return recorded_runtime, descriptor

    def _materialize_interval(
        self,
        profile: AcceptanceProfile,
        input_folder: Path,
        checkpoint_root: Path,
        zero_based_index: int,
        media_runtime: Mapping[str, str],
    ) -> dict[str, object]:
        """一つのintervalをstream copyして実測境界を検証する。"""
        interval = profile.release_intervals[zero_based_index]
        source = profile.input_root / interval.relative_video_path
        _require_source_within_root(profile.input_root, source)
        if not source.is_file():
            raise ValueError("Release interval sourceが存在しません")
        source_snapshot = _source_snapshot(source)
        scenario_id = f"scenario-{zero_based_index + 1:03d}"
        output = input_folder / f"{scenario_id}.mkv"
        checkpoint_path = checkpoint_root / f".{scenario_id}.checkpoint.json"
        pending_checkpoint_path = checkpoint_root / f".{scenario_id}.pending.json"
        restored = self._restore_interval_for_runtime(
            profile,
            zero_based_index,
            source_snapshot,
            output,
            checkpoint_path,
            pending_checkpoint_path,
            media_runtime,
        )
        if restored is not None:
            return restored
        temporary_root = checkpoint_root / (f".{scenario_id}.{uuid4().hex}.tmp")
        _remove_recognized_interval_temporary_roots(checkpoint_root, scenario_id)
        temporary_root.mkdir()
        temporary_output = temporary_root / output.name
        try:
            source_probe = self._media_probe(source)
            source_start = _probe_fraction(source_probe, "start")
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
                    _ffmpeg_time(source_start + interval.end),
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
                    str(temporary_output),
                ]
            )
            clip_probe = self._media_probe(temporary_output)
            clip_start = _probe_fraction(clip_probe, "start")
            clip_duration = _probe_fraction(clip_probe, "duration")
            clip_end_timestamp = _probe_fraction(clip_probe, "end")
            actual_start = clip_start - source_start
            actual_end = clip_end_timestamp - source_start
            duration = actual_end - actual_start
            tolerance = profile.release_boundary_tolerance_seconds
            if (
                duration <= 0
                or duration != clip_duration
                or abs(actual_start - interval.start) > tolerance
                or abs(actual_end - interval.end) > tolerance
                or _probe_streams(source_probe) != _probe_streams(clip_probe)
            ):
                raise ValueError("Release clipの実測境界またはstream保持が不正です")
            if _source_snapshot(source) != source_snapshot:
                raise ValueError("Release interval sourceが処理中に変更されました")
            descriptor = {
                "scenario_id": scenario_id,
                "start": _fraction_record(actual_start),
                "end": _fraction_record(actual_end),
                "duration": _fraction_record(duration),
                "content_sha256": self._content_digester(temporary_output),
                "stream_count": len(_probe_streams(clip_probe)),
            }
            _fsync_file_and_parent(temporary_output)
            if output.exists() and output.is_dir() and not output.is_symlink():
                raise ValueError("Release interval outputがdirectoryです")
            checkpoint = {
                "schema": _INTERVAL_SCHEMA,
                "profile_digest": profile.profile_digest,
                "interval_index": zero_based_index,
                "media_runtime_identity": dict(media_runtime),
                "source_snapshot": source_snapshot,
                "interval": _interval_record(interval),
                "descriptor": descriptor,
            }
            write_atomic_json(pending_checkpoint_path, checkpoint)
            temporary_output.replace(output)
            _fsync_file_and_parent(output)
            _promote_checkpoint(pending_checkpoint_path, checkpoint_path)
            return descriptor
        finally:
            shutil.rmtree(temporary_root, ignore_errors=True)

    def _restore_interval_for_runtime(
        self,
        profile: AcceptanceProfile,
        zero_based_index: int,
        source_snapshot: dict[str, int],
        output: Path,
        checkpoint_path: Path,
        pending_checkpoint_path: Path,
        media_runtime: Mapping[str, str],
    ) -> dict[str, object] | None:
        """現在runtimeのpendingまたは確定checkpointを復元する。"""
        for candidate_path in (pending_checkpoint_path, checkpoint_path):
            restored = self._restore_interval(
                profile,
                zero_based_index,
                source_snapshot,
                output,
                candidate_path,
                media_runtime,
            )
            if restored is None:
                continue
            if candidate_path == pending_checkpoint_path:
                _promote_checkpoint(pending_checkpoint_path, checkpoint_path)
            return restored
        return None

    def _restore_recorded_interval(
        self,
        profile: AcceptanceProfile,
        zero_based_index: int,
        source_snapshot: dict[str, int],
        output: Path,
        checkpoint_path: Path,
        pending_checkpoint_path: Path,
    ) -> tuple[dict[str, str], dict[str, object]] | None:
        """記録済みruntimeを使い、最新の健全なinterval checkpointを復元する。"""
        for candidate_path in (pending_checkpoint_path, checkpoint_path):
            checkpoint = _read_checkpoint_object(candidate_path)
            if checkpoint is None:
                continue
            runtime = parse_media_runtime_identity_record(
                checkpoint.get("media_runtime_identity")
            )
            if runtime is None:
                continue
            restored = self._restore_interval(
                profile,
                zero_based_index,
                source_snapshot,
                output,
                candidate_path,
                runtime,
            )
            if restored is None:
                continue
            if candidate_path == pending_checkpoint_path:
                _promote_checkpoint(pending_checkpoint_path, checkpoint_path)
            return runtime, restored
        return None

    def _restore_interval(
        self,
        profile: AcceptanceProfile,
        zero_based_index: int,
        source_snapshot: dict[str, int],
        output: Path,
        checkpoint_path: Path,
        media_runtime: Mapping[str, str],
    ) -> dict[str, object] | None:
        """一つの確定済みintervalをsourceとartifact検証後に復元する。"""
        checkpoint = _read_checkpoint_object(checkpoint_path)
        if checkpoint is None:
            return None
        interval = profile.release_intervals[zero_based_index]
        descriptor = checkpoint.get("descriptor")
        typed_descriptor = (
            cast(dict[str, object], descriptor) if isinstance(descriptor, dict) else {}
        )
        valid = (
            checkpoint.get("schema") == _INTERVAL_SCHEMA
            and checkpoint.get("profile_digest") == profile.profile_digest
            and checkpoint.get("interval_index") == zero_based_index
            and checkpoint.get("media_runtime_identity") == media_runtime
            and checkpoint.get("source_snapshot") == source_snapshot
            and checkpoint.get("interval") == _interval_record(interval)
            and isinstance(descriptor, dict)
            and typed_descriptor.get("scenario_id")
            == f"scenario-{zero_based_index + 1:03d}"
            and _is_regular_file(output)
        )
        if valid:
            try:
                valid = typed_descriptor.get(
                    "content_sha256"
                ) == self._content_digester(output)
                _validate_clip_descriptor(
                    profile,
                    zero_based_index,
                    typed_descriptor,
                )
            except (
                FileNotFoundError,
                IsADirectoryError,
                NotADirectoryError,
                TypeError,
                ValueError,
            ):
                valid = False
        if not valid:
            return None
        return typed_descriptor

    def _restore_existing(
        self,
        profile: AcceptanceProfile,
        input_folder: Path,
        manifest: dict[str, object],
        source_snapshot: str,
    ) -> dict[str, object]:
        """同じprofileの確定済み匿名clipだけをresume用に再利用する。"""
        descriptor = manifest.get("descriptor")
        if (
            manifest.get("schema") != _MATERIALIZATION_SCHEMA
            or manifest.get("profile_digest") != profile.profile_digest
            or not isinstance(descriptor, dict)
            or input_folder.is_symlink()
            or not input_folder.is_dir()
        ):
            raise ValueError("Release suite stateがprofileと一致しません")
        _validate_suite_descriptor(profile, descriptor)
        if descriptor.get("source_snapshot_fingerprint") != source_snapshot:
            raise ValueError("Release suite sourceが変更されています")
        clips = descriptor.get("clips")
        if not isinstance(clips, list) or len(clips) != len(profile.release_intervals):
            raise ValueError("Release suite clip manifestが不正です")
        expected_names = tuple(
            f"scenario-{index:03d}.mkv"
            for index in range(1, len(profile.release_intervals) + 1)
        )
        actual_names = tuple(
            path.relative_to(input_folder).as_posix()
            for path in discover_video_paths(input_folder, recursive=True)
        )
        if actual_names != expected_names:
            raise ValueError("Release suite匿名inputが変更されています")
        for index, item in enumerate(clips, start=1):
            if not isinstance(item, dict):
                raise ValueError("Release suite clip manifestが不正です")
            path = input_folder / f"scenario-{index:03d}.mkv"
            if not path.is_file() or item.get(
                "content_sha256"
            ) != self._content_digester(path):
                raise ValueError("Release suite clipが変更されています")
        _validate_total_duration(profile, descriptor)
        return cast(dict[str, object], descriptor)


def _prepare_materialization_context(
    profile: AcceptanceProfile,
    input_folder: Path,
    checkpoint_root: Path,
    context_path: Path,
    media_runtime: dict[str, str],
) -> None:
    """未完成interval群を一つのMedia Runtime identityへ固定する。"""
    expected = {
        "schema": _CONTEXT_SCHEMA,
        "profile_digest": profile.profile_digest,
        "media_runtime_identity": media_runtime,
    }
    if context_path.is_symlink() or (
        context_path.exists() and not context_path.is_file()
    ):
        raise ValueError(
            "Release materialization contextが不正です。--reset-suiteが必要です"
        )
    try:
        context = read_json_object(context_path)
    except ValueError:
        context = None
        context_is_trusted = False
    else:
        context_is_trusted = context == expected
    if context_is_trusted:
        return
    for path in (input_folder, checkpoint_root):
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            raise ValueError("Release suite workが不正です。--reset-suiteが必要です")
    write_atomic_json(context_path, expected)


def _source_snapshot(path: Path) -> dict[str, int]:
    """利用者合意済みのsizeとmtimeだけをsource互換性へ使う。"""
    stat = path.stat()
    return {
        "size_bytes": stat.st_size,
        "modified_at_ns": stat.st_mtime_ns,
    }


def _interval_record(interval: ReleaseInterval) -> dict[str, object]:
    """private pathを含めずintervalの意味入力を返す。"""
    return {
        "start": _fraction_record(interval.start),
        "end": _fraction_record(interval.end),
        "scenario_role": interval.scenario_role,
    }


def _remove_recognized_interval_temporary_roots(
    checkpoint_root: Path,
    scenario_id: str,
) -> None:
    """同じintervalが残したUUID形式の未確定一時directoryだけを除く。"""
    prefix = f".{scenario_id}."
    suffix = ".tmp"
    for path in checkpoint_root.iterdir():
        name = path.name
        if not name.startswith(prefix) or not name.endswith(suffix):
            continue
        token = name[len(prefix) : -len(suffix)]
        if len(token) == 32 and all(
            character in "0123456789abcdef" for character in token
        ):
            try:
                mode = path.lstat().st_mode
            except (FileNotFoundError, NotADirectoryError):
                continue
            if stat.S_ISLNK(mode) or stat.S_ISREG(mode):
                path.unlink()
            elif stat.S_ISDIR(mode):
                shutil.rmtree(path, ignore_errors=False)
            else:
                raise ValueError("Release suite一時clip pathが通常fileではありません")


def _is_regular_file(path: Path) -> bool:
    """欠損だけをmissとし、access障害を破損扱いにしない。"""
    try:
        mode = path.lstat().st_mode
    except (FileNotFoundError, NotADirectoryError):
        return False
    return stat.S_ISREG(mode)


def _read_checkpoint_object(path: Path) -> dict[str, object] | None:
    """通常fileのcheckpointだけを読み、local corruptionをmissへ変換する。"""
    if not _is_regular_file(path):
        return None
    try:
        return read_json_object(path)
    except ValueError:
        return None


def _promote_checkpoint(pending_path: Path, checkpoint_path: Path) -> None:
    """artifact検証済みpending checkpointをatomicにcommit markerへ昇格する。"""
    if not _is_regular_file(pending_path):
        raise ValueError("Release interval pending checkpointが不正です")
    pending_path.replace(checkpoint_path)
    descriptor = os.open(checkpoint_path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_anonymous_video_names(
    input_folder: Path,
    interval_count: int,
) -> None:
    """確定対象以外の対応videoが混入していないことを検証する。"""
    expected = tuple(
        f"scenario-{index:03d}.mkv" for index in range(1, interval_count + 1)
    )
    actual = tuple(
        path.relative_to(input_folder).as_posix()
        for path in discover_video_paths(input_folder, recursive=True)
    )
    if actual != expected:
        raise ValueError("Release suite匿名inputが変更されています")


def _fsync_file_and_parent(path: Path) -> None:
    """interval manifestを確定する前にclip bytesとdirectory entryをflushする。"""
    with path.open("rb") as file:
        os.fsync(file.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _suite_descriptor(
    clips: tuple[dict[str, object], ...],
    source_snapshot: str,
) -> dict[str, object]:
    total = sum(
        (_record_fraction(item["duration"]) for item in clips),
        start=Fraction(0),
    )
    canonical = json.dumps(clips, sort_keys=True, separators=(",", ":")).encode()
    return {
        "suite_fingerprint": hashlib.sha256(canonical).hexdigest(),
        "source_snapshot_fingerprint": source_snapshot,
        "scenario_count": len(clips),
        "total_duration": _fraction_record(total),
        "clips": list(clips),
    }


def _manifest_descriptor_is_valid(
    profile: AcceptanceProfile,
    manifest: Mapping[str, object],
) -> bool:
    """終端manifestのidentityとdescriptorが自己整合するか返す。"""
    descriptor = manifest.get("descriptor")
    if (
        manifest.get("schema") != _MATERIALIZATION_SCHEMA
        or manifest.get("profile_digest") != profile.profile_digest
        or not isinstance(descriptor, dict)
    ):
        return False
    try:
        _validate_suite_descriptor(profile, descriptor)
    except (TypeError, ValueError):
        return False
    return True


def _validate_suite_descriptor(
    profile: AcceptanceProfile,
    descriptor: Mapping[str, object],
) -> None:
    """Release suite descriptorの全導出値とclip関係を検証する。"""
    expected_keys = {
        "suite_fingerprint",
        "source_snapshot_fingerprint",
        "scenario_count",
        "total_duration",
        "clips",
    }
    clips = descriptor.get("clips")
    if (
        set(descriptor) != expected_keys
        or descriptor.get("scenario_count") != len(profile.release_intervals)
        or not _is_sha256(descriptor.get("source_snapshot_fingerprint"))
        or not isinstance(clips, list)
        or len(clips) != len(profile.release_intervals)
        or any(not isinstance(item, dict) for item in clips)
    ):
        raise ValueError("Release suite clip manifestが不正です")
    typed_clips = tuple(cast(dict[str, object], item) for item in clips)
    for zero_based_index, clip in enumerate(typed_clips):
        _validate_clip_descriptor(profile, zero_based_index, clip)
    source_snapshot = cast(str, descriptor["source_snapshot_fingerprint"])
    if dict(descriptor) != _suite_descriptor(typed_clips, source_snapshot):
        raise ValueError("Release suite descriptorの導出値が不正です")
    _validate_total_duration(profile, descriptor)


def _validate_clip_descriptor(
    profile: AcceptanceProfile,
    zero_based_index: int,
    descriptor: Mapping[str, object],
) -> None:
    """一つのclip descriptorをprofile区間と照合する。"""
    expected_keys = {
        "scenario_id",
        "start",
        "end",
        "duration",
        "content_sha256",
        "stream_count",
    }
    interval = profile.release_intervals[zero_based_index]
    start = _record_fraction(descriptor.get("start"))
    end = _record_fraction(descriptor.get("end"))
    duration = _record_fraction(descriptor.get("duration"))
    stream_count = descriptor.get("stream_count")
    if (
        set(descriptor) != expected_keys
        or descriptor.get("scenario_id") != f"scenario-{zero_based_index + 1:03d}"
        or not _is_sha256(descriptor.get("content_sha256"))
        or not isinstance(stream_count, int)
        or isinstance(stream_count, bool)
        or stream_count < 1
        or start < 0
        or end <= start
        or duration != end - start
        or abs(start - interval.start) > profile.release_boundary_tolerance_seconds
        or abs(end - interval.end) > profile.release_boundary_tolerance_seconds
    ):
        raise ValueError("Release suite clip descriptorが不正です")


def _validate_total_duration(
    profile: AcceptanceProfile,
    descriptor: Mapping[str, object],
) -> None:
    """実測clip合計がprofileのrelease duration境界内であることを検証する。"""
    measured = _record_fraction(descriptor.get("total_duration"))
    if (
        abs(measured - profile.release_expected_total_duration)
        > profile.release_boundary_tolerance_seconds
    ):
        raise ValueError("Release suiteの実測合計durationが不正です")


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
    """container差を吸収したstart、経過duration、absolute endを返す。"""
    try:
        process = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                (
                    "format=format_name,start_time,duration:"
                    "stream=codec_type,codec_name,start_time,duration"
                ),
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
    try:
        format_start = Fraction(str(format_value.get("start_time", "0")))
        format_duration = Fraction(str(format_value["duration"]))
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        raise ValueError("FFprobe format timingが不正です") from None
    format_names = {
        item.strip().casefold()
        for item in str(format_value.get("format_name", "")).split(",")
        if item.strip()
    }
    streams: list[tuple[str, str]] = []
    stream_end_timestamps: list[Fraction] = []
    for item in streams_value:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("codec_type"), str)
            or not isinstance(item.get("codec_name"), str)
        ):
            raise ValueError("FFprobe streamが不正です")
        streams.append((item["codec_type"], item["codec_name"]))
        stream_duration_value = item.get("duration")
        if stream_duration_value is None:
            continue
        try:
            stream_start = Fraction(str(item.get("start_time", format_start)))
            stream_duration = Fraction(str(stream_duration_value))
        except (TypeError, ValueError, ZeroDivisionError):
            raise ValueError("FFprobe stream timingが不正です") from None
        if stream_duration > 0:
            stream_end_timestamps.append(stream_start + stream_duration)
    if stream_end_timestamps:
        end_timestamp = max(stream_end_timestamps)
        elapsed_duration = end_timestamp - format_start
    elif format_names & {"matroska", "webm"} and format_duration > format_start:
        # Matroska demuxerは非0 PTSでAVFormatContext.durationをendとして返す。
        end_timestamp = format_duration
        elapsed_duration = end_timestamp - format_start
    else:
        elapsed_duration = format_duration
        end_timestamp = format_start + elapsed_duration
    if elapsed_duration <= 0 or end_timestamp <= format_start:
        raise ValueError("FFprobe media timingが不正です")
    return {
        "start": format_start,
        "duration": elapsed_duration,
        "end": end_timestamp,
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
    if (
        not isinstance(numerator, int)
        or isinstance(numerator, bool)
        or not isinstance(denominator, int)
        or isinstance(denominator, bool)
        or denominator == 0
    ):
        raise ValueError("Duration recordが不正です")
    return Fraction(numerator, denominator)


def _read_materialization_manifest(path: Path) -> dict[str, object] | None:
    """破損した通常file manifestを捨て、unit checkpointから再構築可能にする。"""
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError("Release materialization manifestが通常fileではありません")
    try:
        return read_json_object(path)
    except ValueError:
        path.unlink(missing_ok=True)
        return None


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
