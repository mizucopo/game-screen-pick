"""長時間acceptance開始前のartifact容量preflight。"""

import shutil
from collections.abc import Callable
from pathlib import Path

from ..services.discover_video_paths import discover_video_paths
from .acceptance_profile import AcceptanceProfile

DiskUsageProbe = Callable[[Path], tuple[int, int, int]]

PERSISTENT_CACHE_BUDGET_BYTES = 64 * 1024**3
PEAK_ADDITIONAL_BUDGET_BYTES = 96 * 1024**3
REQUIRED_ARTIFACT_CAPACITY_BYTES = (
    PERSISTENT_CACHE_BUDGET_BYTES + PEAK_ADDITIONAL_BUDGET_BYTES
)


def preflight_acceptance_storage(
    profile: AcceptanceProfile,
    *,
    disk_usage_probe: DiskUsageProbe = shutil.disk_usage,
) -> dict[str, object]:
    """input規模とartifact空き容量を測りbudget未満なら開始を拒否する。"""
    video_paths = discover_video_paths(profile.input_root, recursive=True)
    input_video_bytes = sum(path.stat().st_size for path in video_paths)
    _total, _used, available = disk_usage_probe(profile.artifact_root)
    if available < REQUIRED_ARTIFACT_CAPACITY_BYTES:
        raise ValueError(
            "Acceptance artifact容量が不足しています: "
            f"required={REQUIRED_ARTIFACT_CAPACITY_BYTES}, available={available}"
        )
    return {
        "input_video_bytes": input_video_bytes,
        "input_video_count": len(video_paths),
        "artifact_available_bytes": available,
        "required_artifact_capacity_bytes": REQUIRED_ARTIFACT_CAPACITY_BYTES,
        "persistent_cache_budget_bytes": PERSISTENT_CACHE_BUDGET_BYTES,
        "peak_additional_budget_bytes": PEAK_ADDITIONAL_BUDGET_BYTES,
    }
