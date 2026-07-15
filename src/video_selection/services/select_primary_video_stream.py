"""Primary Video Streamを決定する。"""

from ..models.media_probe import MediaProbe
from ..models.media_stream import MediaStream


def select_primary_video_stream(probe: MediaProbe) -> MediaStream:
    """coverを除外しdefault、stream indexの順で表示映像を選ぶ。"""
    candidates = tuple(
        stream
        for stream in probe.streams
        if stream.kind == "video" and not stream.is_attached_picture
    )
    if not candidates:
        msg = "Primary Video Streamとなる表示映像streamがありません"
        raise ValueError(msg)
    return min(candidates, key=lambda stream: (not stream.is_default, stream.index))
