"""full-scale sourceを匿名symlink Video Setへmaterializeする。"""

import hashlib
import json
import shutil
from collections.abc import Callable, Mapping
from fractions import Fraction
from pathlib import Path
from typing import cast

from ..services.discover_video_paths import discover_video_paths
from .acceptance_profile import AcceptanceProfile
from .atomic_json import read_json_object, write_atomic_json
from .release_suite_materializer import _probe_media

MediaProbe = Callable[[Path], Mapping[str, object]]

_MATERIALIZATION_SCHEMA = "game-screen-pick/full-materialization@2.0.0"


class FullSuiteMaterializer:
    """full-scale sourceをcopyせず匿名・cache分離されたinput viewへ固定する。"""

    def __init__(self, *, media_probe: MediaProbe | None = None) -> None:
        self._media_probe = media_probe or _probe_media

    def materialize(
        self,
        profile: AcceptanceProfile,
        suite_root: Path,
    ) -> tuple[Path, dict[str, object]]:
        """匿名symlink input folderとpath非依存descriptorを返す。"""
        sources = discover_video_paths(profile.input_root, recursive=True)
        if len(sources) != profile.full_expected_video_count:
            raise ValueError("Full suiteのVideo countがprofile期待値と一致しません")
        source_snapshot = _source_snapshot_fingerprint(sources)
        work_root = suite_root / "work"
        input_folder = work_root / "input"
        manifest_path = work_root / "full-materialization.json"
        existing = read_json_object(manifest_path)
        if existing is not None:
            return input_folder, _restore_existing(
                profile,
                input_folder,
                sources,
                source_snapshot,
                existing,
            )
        if input_folder.exists():
            raise ValueError("Full suite workが未確定です。--reset-suiteが必要です")
        input_folder.mkdir(parents=True)
        try:
            durations: list[Fraction] = []
            anonymous_names: list[str] = []
            for index, source in enumerate(sources, start=1):
                name = f"scenario-{index:03d}{source.suffix.casefold()}"
                (input_folder / name).symlink_to(source.resolve(strict=True))
                anonymous_names.append(name)
                durations.append(_probe_elapsed_duration(self._media_probe(source)))
            total_duration = sum(durations, start=Fraction(0))
            if (
                abs(total_duration - profile.full_expected_total_duration)
                > profile.full_duration_tolerance_seconds
            ):
                msg = "Full suiteの実測durationがprofile期待値と一致しません"
                raise ValueError(msg)
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
                    "descriptor": descriptor,
                },
            )
            return input_folder, descriptor
        except BaseException:
            shutil.rmtree(input_folder, ignore_errors=True)
            raise


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
    names = descriptor.get("anonymous_video_names")
    if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
        raise ValueError("Full suite匿名input manifestが不正です")
    paths = tuple(input_folder / name for name in names)
    if len(paths) != profile.full_expected_video_count or len(paths) != len(sources):
        raise ValueError("Full suite匿名inputが変更されています")
    if not input_folder.is_dir():
        raise ValueError("Full suite匿名inputが変更されています")
    actual_names = tuple(
        path.relative_to(input_folder).as_posix()
        for path in discover_video_paths(input_folder, recursive=True)
    )
    if actual_names != tuple(names):
        raise ValueError("Full suite匿名inputが変更されています")
    for path, source in zip(paths, sources, strict=True):
        try:
            target_matches = (
                path.is_symlink()
                and path.is_file()
                and path.resolve(strict=True) == source.resolve(strict=True)
            )
        except OSError:
            target_matches = False
        if not target_matches:
            raise ValueError("Full suite匿名inputが変更されています")
    return cast(dict[str, object], descriptor)


def _source_snapshot_fingerprint(sources: tuple[Path, ...]) -> str:
    records: list[dict[str, object]] = []
    for source in sources:
        stat = source.stat()
        records.append(
            {
                "device": stat.st_dev,
                "inode": stat.st_ino,
                "size_bytes": stat.st_size,
                "modified_at_ns": stat.st_mtime_ns,
                "changed_at_ns": stat.st_ctime_ns,
                "suffix": source.suffix.casefold(),
            }
        )
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _probe_elapsed_duration(value: Mapping[str, object]) -> Fraction:
    start = value.get("start")
    duration = value.get("duration")
    if (
        not isinstance(start, Fraction)
        or not isinstance(duration, Fraction)
        or duration <= start
    ):
        raise ValueError("Full suite media durationが不正です")
    return duration - start


def _fraction_record(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}
