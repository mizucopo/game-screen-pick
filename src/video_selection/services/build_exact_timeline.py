"""decoded frame timingからexact timelineを構築する。"""

from fractions import Fraction
from itertools import pairwise

from ..models.media_stream import MediaStream
from ..models.timeline_segment import TimelineSegment
from ..models.video_duration import VideoDuration
from ..models.video_timeline import VideoTimeline
from .build_video_entity_id import build_video_entity_id

_SEGMENT_ID_ALGORITHM = "timeline-segment-id-v1"


def build_exact_timeline(
    *,
    video_fingerprint: str,
    stream: MediaStream,
    origin_pts: int,
    last_frame_pts: int,
    last_frame_duration_ts: int | None,
    scene_pts: tuple[int, ...],
) -> VideoTimeline:
    """frame終端を優先しscene境界でgapless segmentを作る。"""
    if stream.time_base is None:
        msg = "Primary Video Streamにexact time baseが必要です"
        raise ValueError(msg)
    end_pts = _resolve_end_pts(
        stream,
        last_frame_pts,
        last_frame_duration_ts,
    )
    duration_seconds = Fraction(end_pts - origin_pts) * stream.time_base
    duration = VideoDuration(duration_seconds)
    scene_times = {
        Fraction(scene_pts_value - origin_pts) * stream.time_base
        for scene_pts_value in scene_pts
    }
    boundaries = (
        Fraction(0),
        *sorted(
            scene_time
            for scene_time in scene_times
            if 0 < scene_time < duration.seconds
        ),
        duration.seconds,
    )
    segments = tuple(
        TimelineSegment(
            identifier=build_video_entity_id(
                "seg",
                _SEGMENT_ID_ALGORITHM,
                video_fingerprint,
                start,
                end,
            ),
            start=start,
            end=end,
        )
        for start, end in pairwise(boundaries)
    )
    return VideoTimeline(
        origin_pts=origin_pts,
        time_base=stream.time_base,
        duration=duration,
        segments=segments,
    )


def _resolve_end_pts(
    stream: MediaStream,
    last_frame_pts: int,
    last_frame_duration_ts: int | None,
) -> int:
    if last_frame_duration_ts is not None and last_frame_duration_ts > 0:
        return last_frame_pts + last_frame_duration_ts
    if (
        stream.start_pts is not None
        and stream.duration_ts is not None
        and stream.duration_ts > 0
    ):
        return stream.start_pts + stream.duration_ts
    msg = "exactな正のVideo Duration終端を得られません"
    raise ValueError(msg)
