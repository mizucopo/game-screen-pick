"""Acceptance sourceの軽量snapshot fingerprintを構築する。"""

import hashlib
import json
from pathlib import Path

from ..services.discover_video_paths import discover_video_paths
from .acceptance_profile import AcceptanceProfile


def acceptance_source_snapshot_fingerprint(
    profile: AcceptanceProfile,
    suite: str,
) -> str:
    """suiteが参照するsourceのsize・mtime・suffixをpathなしで識別する。"""
    return source_snapshot_fingerprint(acceptance_source_paths(profile, suite))


def source_snapshot_fingerprint(sources: tuple[Path, ...]) -> str:
    """順序付きsourceのstat snapshotをpathなしで識別する。"""
    records: list[dict[str, object]] = []
    for source in sources:
        stat = source.stat()
        records.append(
            {
                "size_bytes": stat.st_size,
                "modified_at_ns": stat.st_mtime_ns,
                "suffix": source.suffix.casefold(),
            }
        )
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def acceptance_source_paths(
    profile: AcceptanceProfile,
    suite: str,
) -> tuple[Path, ...]:
    """suite順のsource pathを境界検証後に返す。"""
    if suite == "full":
        return discover_video_paths(profile.input_root, recursive=True)
    if suite != "release":
        raise ValueError("Acceptance suiteが不正です")
    sources = tuple(
        profile.input_root / interval.relative_video_path
        for interval in profile.release_intervals
    )
    for source in sources:
        try:
            source.resolve(strict=True).relative_to(
                profile.input_root.resolve(strict=True)
            )
        except (OSError, ValueError):
            raise ValueError(
                "Release interval sourceがinput root外または未作成です"
            ) from None
        if not source.is_file():
            raise ValueError("Release interval sourceが存在しません")
    return sources
