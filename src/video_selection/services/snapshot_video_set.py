"""Video Setのcontent snapshotを構築する。"""

from ..models.video_set import VideoSet


def snapshot_video_set(video_set: VideoSet) -> tuple[dict[str, str], ...]:
    """Video Orderを保ったrelative pathとcontent digestを返す。"""
    return tuple(
        {
            "path": source.relative_path,
            "sha256": source.fingerprint,
        }
        for source in video_set.sources
    )
