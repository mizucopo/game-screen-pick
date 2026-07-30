"""full-scale sourceを匿名symlink Video Setへmaterializeする。"""

import os
import stat
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
from .release_suite_materializer import _probe_media
from .source_snapshot_fingerprint import (
    acceptance_source_paths,
    source_snapshot_fingerprint,
)
from .suite_owned_directory import validate_suite_owned_directory_chain

MediaProbe = Callable[[Path], Mapping[str, object]]

_MATERIALIZATION_SCHEMA = "game-screen-pick/full-materialization@3.0.0"
_SOURCE_SCHEMA = "game-screen-pick/full-source-checkpoint@2.0.0"
_CONTEXT_SCHEMA = "game-screen-pick/full-materialization-context@2.0.0"


class FullSuiteMaterializer:
    """full-scale sourceをcopyせず匿名・cache分離されたinput viewへ固定する。"""

    def __init__(
        self,
        *,
        media_probe: MediaProbe | None = None,
        media_runtime_probe: MediaRuntimeProbe = probe_media_runtime_identity,
    ) -> None:
        self._media_probe = media_probe or _probe_media
        self._media_runtime_probe = media_runtime_probe

    def materialize(
        self,
        profile: AcceptanceProfile,
        suite_root: Path,
    ) -> tuple[Path, dict[str, object]]:
        """匿名symlink input folderとpath非依存descriptorを返す。"""
        sources = acceptance_source_paths(profile, "full")
        if len(sources) != profile.full_expected_video_count:
            raise ValueError("Full suiteのVideo countがprofile期待値と一致しません")
        source_snapshot = source_snapshot_fingerprint(sources)
        work_root = suite_root / "work"
        input_folder = work_root / "input"
        checkpoint_root = work_root / "source-checkpoints"
        manifest_path = work_root / "full-materialization.json"
        context_path = work_root / "full-materialization-context.json"
        validate_suite_owned_directory_chain(
            suite_root,
            work_root,
            input_folder,
            suite_label="Full",
        )
        existing = _read_materialization_manifest(manifest_path)
        if existing is not None and _manifest_descriptor_is_valid(
            profile,
            sources,
            source_snapshot,
            existing,
        ):
            return input_folder, _restore_existing(
                profile,
                input_folder,
                sources,
                source_snapshot,
                existing,
            )
        recovered = _restore_completed_from_checkpoints(
            profile,
            input_folder,
            checkpoint_root,
            sources,
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
            raise ValueError("Full suite workが不正です。--reset-suiteが必要です")
        input_folder.mkdir(parents=True, exist_ok=True)
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        durations: list[Fraction] = []
        anonymous_names: list[str] = []
        for index, source in enumerate(sources, start=1):
            name, duration = self._materialize_source(
                profile,
                input_folder,
                checkpoint_root,
                source,
                index,
                media_runtime,
            )
            anonymous_names.append(name)
            durations.append(duration)
        total_duration = sum(durations, start=Fraction(0))
        if (
            abs(total_duration - profile.full_expected_total_duration)
            > profile.full_duration_tolerance_seconds
        ):
            msg = "Full suiteの実測durationがprofile期待値と一致しません"
            raise ValueError(msg)
        _validate_source_snapshot(sources, source_snapshot)
        _validate_anonymous_video_names(input_folder, tuple(anonymous_names))
        descriptor = {
            "source_snapshot_fingerprint": source_snapshot,
            "scenario_count": len(sources),
            "total_duration": _fraction_record(total_duration),
            "anonymous_video_names": anonymous_names,
        }
        write_atomic_json(
            manifest_path,
            {
                "schema": _MATERIALIZATION_SCHEMA,
                "profile_digest": profile.profile_digest,
                "media_runtime_identity": media_runtime,
                "descriptor": descriptor,
            },
        )
        return input_folder, descriptor

    def _materialize_source(
        self,
        profile: AcceptanceProfile,
        input_folder: Path,
        checkpoint_root: Path,
        source: Path,
        one_based_index: int,
        media_runtime: Mapping[str, str],
    ) -> tuple[str, Fraction]:
        """一つのsource symlinkとdurationをcommit marker付きで確定する。"""
        name = f"scenario-{one_based_index:03d}{source.suffix.casefold()}"
        anonymous_path = input_folder / name
        checkpoint_name = f".scenario-{one_based_index:03d}.checkpoint.json"
        checkpoint_path = checkpoint_root / checkpoint_name
        pending_checkpoint_path = (
            checkpoint_root / f".scenario-{one_based_index:03d}.pending.json"
        )
        source_record = _source_record(source)
        restored = _restore_source_for_runtime(
            profile,
            source,
            one_based_index,
            name,
            source_record,
            anonymous_path,
            checkpoint_path,
            pending_checkpoint_path,
            media_runtime,
        )
        if restored is not None:
            return name, restored
        temporary_link = input_folder / (f".{anonymous_path.name}.{uuid4().hex}.tmp")
        _remove_recognized_source_temporary_links(input_folder, anonymous_path.name)
        try:
            duration = _probe_elapsed_duration(self._media_probe(source))
            if _source_record(source) != source_record:
                raise ValueError("Full suite sourceがmaterialize中に変更されました")
            checkpoint = {
                "schema": _SOURCE_SCHEMA,
                "profile_digest": profile.profile_digest,
                "source_index": one_based_index,
                "media_runtime_identity": dict(media_runtime),
                "source_snapshot": source_record,
                "anonymous_name": name,
                "duration": _fraction_record(duration),
            }
            write_atomic_json(pending_checkpoint_path, checkpoint)
            if not _source_link_matches(anonymous_path, source):
                temporary_link.symlink_to(source.resolve(strict=True))
                if (
                    anonymous_path.exists()
                    and anonymous_path.is_dir()
                    and not anonymous_path.is_symlink()
                ):
                    raise ValueError("Full suite匿名inputがdirectoryです")
                temporary_link.replace(anonymous_path)
                _fsync_directory(input_folder)
            _promote_checkpoint(pending_checkpoint_path, checkpoint_path)
            return name, duration
        finally:
            temporary_link.unlink(missing_ok=True)


def _prepare_materialization_context(
    profile: AcceptanceProfile,
    input_folder: Path,
    checkpoint_root: Path,
    context_path: Path,
    media_runtime: dict[str, str],
) -> None:
    """未完成source群を一つのMedia Runtime identityへ固定する。"""
    expected = {
        "schema": _CONTEXT_SCHEMA,
        "profile_digest": profile.profile_digest,
        "media_runtime_identity": media_runtime,
    }
    if context_path.is_symlink() or (
        context_path.exists() and not context_path.is_file()
    ):
        raise ValueError(
            "Full materialization contextが不正です。--reset-suiteが必要です"
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
            raise ValueError("Full suite workが不正です。--reset-suiteが必要です")
    write_atomic_json(context_path, expected)


def _restore_existing(
    profile: AcceptanceProfile,
    input_folder: Path,
    sources: tuple[Path, ...],
    source_snapshot: str,
    manifest: dict[str, object],
) -> dict[str, object]:
    descriptor = manifest.get("descriptor")
    if (
        manifest.get("schema") != _MATERIALIZATION_SCHEMA
        or manifest.get("profile_digest") != profile.profile_digest
        or not isinstance(descriptor, dict)
        or descriptor.get("source_snapshot_fingerprint") != source_snapshot
    ):
        raise ValueError("Full suite stateがprofileまたはsourceと一致しません")
    _validate_suite_descriptor(
        profile,
        sources,
        source_snapshot,
        descriptor,
    )
    names = descriptor.get("anonymous_video_names")
    if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
        raise ValueError("Full suite匿名input manifestが不正です")
    paths = tuple(input_folder / name for name in names)
    if len(paths) != profile.full_expected_video_count or len(paths) != len(sources):
        raise ValueError("Full suite匿名inputが変更されています")
    if input_folder.is_symlink() or not input_folder.is_dir():
        raise ValueError("Full suite匿名inputが変更されています")
    actual_names = tuple(
        path.relative_to(input_folder).as_posix()
        for path in discover_video_paths(input_folder, recursive=True)
    )
    if actual_names != tuple(names):
        raise ValueError("Full suite匿名inputが変更されています")
    for path, source in zip(paths, sources, strict=True):
        if not _source_link_matches(path, source):
            raise ValueError("Full suite匿名inputが変更されています")
    _validate_source_snapshot(sources, source_snapshot)
    return cast(dict[str, object], descriptor)


def _restore_completed_from_checkpoints(
    profile: AcceptanceProfile,
    input_folder: Path,
    checkpoint_root: Path,
    sources: tuple[Path, ...],
    source_snapshot: str,
) -> tuple[dict[str, str], dict[str, object]] | None:
    """全sourceが同じ記録済みruntimeで確定済みなら終端manifestを復元する。"""
    if (
        input_folder.is_symlink()
        or checkpoint_root.is_symlink()
        or not input_folder.is_dir()
        or not checkpoint_root.is_dir()
    ):
        return None
    recorded_runtime: dict[str, str] | None = None
    durations: list[Fraction] = []
    names: list[str] = []
    for index, source in enumerate(sources, start=1):
        name = f"scenario-{index:03d}{source.suffix.casefold()}"
        checkpoint_path = checkpoint_root / f".scenario-{index:03d}.checkpoint.json"
        pending_checkpoint_path = (
            checkpoint_root / f".scenario-{index:03d}.pending.json"
        )
        restored_checkpoint = _restore_recorded_source(
            profile,
            source,
            index,
            name,
            _source_record(source),
            input_folder / name,
            checkpoint_path,
            pending_checkpoint_path,
        )
        if restored_checkpoint is None:
            return None
        runtime, restored = restored_checkpoint
        if recorded_runtime is not None and runtime != recorded_runtime:
            return None
        recorded_runtime = runtime
        names.append(name)
        durations.append(restored)
    if recorded_runtime is None:
        return None
    total_duration = sum(durations, start=Fraction(0))
    if (
        abs(total_duration - profile.full_expected_total_duration)
        > profile.full_duration_tolerance_seconds
    ):
        raise ValueError("Full suiteの実測durationがprofile期待値と一致しません")
    _validate_source_snapshot(sources, source_snapshot)
    _validate_anonymous_video_names(input_folder, tuple(names))
    return (
        recorded_runtime,
        {
            "source_snapshot_fingerprint": source_snapshot,
            "scenario_count": len(sources),
            "total_duration": _fraction_record(total_duration),
            "anonymous_video_names": names,
        },
    )


def _source_record(source: Path) -> dict[str, int]:
    """source互換性を利用者合意済みのsizeとmtimeだけで記録する。"""
    stat = source.stat()
    return {
        "size_bytes": stat.st_size,
        "modified_at_ns": stat.st_mtime_ns,
    }


def _manifest_descriptor_is_valid(
    profile: AcceptanceProfile,
    sources: tuple[Path, ...],
    source_snapshot: str,
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
    stored_snapshot = descriptor.get("source_snapshot_fingerprint")
    if not _is_sha256(stored_snapshot):
        return False
    if stored_snapshot != source_snapshot:
        # 有効な旧snapshotは破損扱いにせず、restore側でsource変更として拒否する。
        return True
    try:
        _validate_suite_descriptor(
            profile,
            sources,
            source_snapshot,
            descriptor,
        )
    except (TypeError, ValueError):
        return False
    return True


def _validate_suite_descriptor(
    profile: AcceptanceProfile,
    sources: tuple[Path, ...],
    source_snapshot: str,
    descriptor: Mapping[str, object],
) -> None:
    """Full suite descriptorのsource導出値とduration境界を検証する。"""
    expected_names = [
        f"scenario-{index:03d}{source.suffix.casefold()}"
        for index, source in enumerate(sources, start=1)
    ]
    total_duration = _record_fraction(descriptor.get("total_duration"))
    if (
        set(descriptor)
        != {
            "source_snapshot_fingerprint",
            "scenario_count",
            "total_duration",
            "anonymous_video_names",
        }
        or descriptor.get("source_snapshot_fingerprint") != source_snapshot
        or descriptor.get("scenario_count") != len(sources)
        or descriptor.get("scenario_count") != profile.full_expected_video_count
        or descriptor.get("anonymous_video_names") != expected_names
        or total_duration <= 0
        or abs(total_duration - profile.full_expected_total_duration)
        > profile.full_duration_tolerance_seconds
    ):
        raise ValueError("Full suite匿名input manifestが不正です")


def _restore_source_checkpoint(
    profile: AcceptanceProfile,
    source: Path,
    one_based_index: int,
    anonymous_name: str,
    source_record: dict[str, int],
    anonymous_path: Path,
    checkpoint_path: Path,
    media_runtime: Mapping[str, str],
) -> Fraction | None:
    """一つのsource probe checkpointをsymlink検証後に復元する。"""
    checkpoint = _read_checkpoint_object(checkpoint_path)
    if checkpoint is None:
        return None
    target_matches = _source_link_matches(anonymous_path, source)
    valid = (
        checkpoint.get("schema") == _SOURCE_SCHEMA
        and checkpoint.get("profile_digest") == profile.profile_digest
        and checkpoint.get("source_index") == one_based_index
        and checkpoint.get("media_runtime_identity") == media_runtime
        and checkpoint.get("source_snapshot") == source_record
        and checkpoint.get("anonymous_name") == anonymous_name
        and target_matches
    )
    try:
        duration = _record_fraction(checkpoint.get("duration")) if valid else None
    except (TypeError, ValueError, ZeroDivisionError):
        duration = None
    if duration is None or duration <= 0:
        return None
    return duration


def _restore_source_for_runtime(
    profile: AcceptanceProfile,
    source: Path,
    one_based_index: int,
    anonymous_name: str,
    source_record: dict[str, int],
    anonymous_path: Path,
    checkpoint_path: Path,
    pending_checkpoint_path: Path,
    media_runtime: Mapping[str, str],
) -> Fraction | None:
    """現在runtimeのpendingまたは確定source checkpointを復元する。"""
    for candidate_path in (pending_checkpoint_path, checkpoint_path):
        restored = _restore_source_checkpoint(
            profile,
            source,
            one_based_index,
            anonymous_name,
            source_record,
            anonymous_path,
            candidate_path,
            media_runtime,
        )
        if restored is None:
            continue
        if candidate_path == pending_checkpoint_path:
            _promote_checkpoint(pending_checkpoint_path, checkpoint_path)
        return restored
    return None


def _restore_recorded_source(
    profile: AcceptanceProfile,
    source: Path,
    one_based_index: int,
    anonymous_name: str,
    source_record: dict[str, int],
    anonymous_path: Path,
    checkpoint_path: Path,
    pending_checkpoint_path: Path,
) -> tuple[dict[str, str], Fraction] | None:
    """記録済みruntimeを使い、最新の健全なsource checkpointを復元する。"""
    for candidate_path in (pending_checkpoint_path, checkpoint_path):
        checkpoint = _read_checkpoint_object(candidate_path)
        if checkpoint is None:
            continue
        runtime = parse_media_runtime_identity_record(
            checkpoint.get("media_runtime_identity")
        )
        if runtime is None:
            continue
        restored = _restore_source_checkpoint(
            profile,
            source,
            one_based_index,
            anonymous_name,
            source_record,
            anonymous_path,
            candidate_path,
            runtime,
        )
        if restored is None:
            continue
        if candidate_path == pending_checkpoint_path:
            _promote_checkpoint(pending_checkpoint_path, checkpoint_path)
        return runtime, restored
    return None


def _source_link_matches(path: Path, source: Path) -> bool:
    """匿名symlinkが同じsourceを指す場合だけ再利用可能とする。"""
    try:
        mode = path.lstat().st_mode
    except (FileNotFoundError, NotADirectoryError):
        return False
    if not stat.S_ISLNK(mode):
        return False
    try:
        return path.resolve(strict=True) == source.resolve(strict=True)
    except (FileNotFoundError, NotADirectoryError):
        return False


def _remove_recognized_source_temporary_links(
    input_folder: Path,
    anonymous_name: str,
) -> None:
    """同じsourceが残したUUID形式の未確定symlinkだけを除く。"""
    prefix = f".{anonymous_name}."
    suffix = ".tmp"
    for path in input_folder.iterdir():
        name = path.name
        if not name.startswith(prefix) or not name.endswith(suffix):
            continue
        token = name[len(prefix) : -len(suffix)]
        if len(token) != 32 or any(
            character not in "0123456789abcdef" for character in token
        ):
            continue
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.exists():
            raise ValueError("Full suite一時symlink pathが通常fileではありません")


def _fsync_directory(path: Path) -> None:
    """symlinkのatomic replaceをcheckpointより先に永続化する。"""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_checkpoint_object(path: Path) -> dict[str, object] | None:
    """通常fileのcheckpointだけを読み、local corruptionをmissへ変換する。"""
    try:
        mode = path.lstat().st_mode
    except (FileNotFoundError, NotADirectoryError):
        return None
    if not stat.S_ISREG(mode):
        return None
    try:
        return read_json_object(path)
    except ValueError:
        return None


def _promote_checkpoint(pending_path: Path, checkpoint_path: Path) -> None:
    """source検証済みpending checkpointをatomicにcommit markerへ昇格する。"""
    try:
        mode = pending_path.lstat().st_mode
    except (FileNotFoundError, NotADirectoryError):
        raise ValueError("Full source pending checkpointが欠損しています") from None
    if not stat.S_ISREG(mode):
        raise ValueError("Full source pending checkpointが不正です")
    pending_path.replace(checkpoint_path)
    _fsync_directory(checkpoint_path.parent)


def _validate_anonymous_video_names(
    input_folder: Path,
    expected_names: tuple[str, ...],
) -> None:
    """確定対象以外の対応videoが混入していないことを検証する。"""
    actual_names = tuple(
        path.relative_to(input_folder).as_posix()
        for path in discover_video_paths(input_folder, recursive=True)
    )
    if actual_names != expected_names:
        raise ValueError("Full suite匿名inputが変更されています")


def _validate_source_snapshot(
    sources: tuple[Path, ...],
    expected_snapshot: str,
) -> None:
    try:
        current_snapshot = source_snapshot_fingerprint(sources)
    except OSError:
        raise ValueError("Full suite sourceがmaterialize中に変更されました") from None
    if current_snapshot != expected_snapshot:
        raise ValueError("Full suite sourceがmaterialize中に変更されました")


def _probe_elapsed_duration(value: Mapping[str, object]) -> Fraction:
    duration = value.get("duration")
    if not isinstance(duration, Fraction) or duration <= 0:
        raise ValueError("Full suite media durationが不正です")
    return duration


def _fraction_record(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _record_fraction(value: object) -> Fraction:
    if not isinstance(value, dict):
        raise ValueError("Full source duration recordが不正です")
    numerator = value.get("numerator")
    denominator = value.get("denominator")
    if (
        not isinstance(numerator, int)
        or isinstance(numerator, bool)
        or not isinstance(denominator, int)
        or isinstance(denominator, bool)
        or denominator == 0
    ):
        raise ValueError("Full source duration recordが不正です")
    return Fraction(numerator, denominator)


def _read_materialization_manifest(path: Path) -> dict[str, object] | None:
    """破損した通常file manifestを捨て、unit checkpointから再構築可能にする。"""
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError("Full materialization manifestが通常fileではありません")
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
