"""Video Scan resource取得失敗を安全側のsample欠落へ正規化する。"""

from collections.abc import Callable

from ..models.video_scan_resource_sample import VideoScanResourceSample

ResourceSampler = Callable[[], VideoScanResourceSample | None]


def sample_video_scan_resources_safely(
    sampler: ResourceSampler,
) -> VideoScanResourceSample | None:
    """resource取得失敗時にworkerを増やさないためNoneを返す。"""
    try:
        return sampler()
    except Exception:
        return None
