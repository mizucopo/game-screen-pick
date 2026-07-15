"""一つのVideo Sourceのexact timeline。"""

from dataclasses import dataclass
from fractions import Fraction

from .timeline_segment import TimelineSegment
from .video_duration import VideoDuration


@dataclass(frozen=True)
class VideoTimeline:
    """origin、duration、gapless segment列を保持する。"""

    origin_pts: int
    time_base: Fraction
    duration: VideoDuration
    segments: tuple[TimelineSegment, ...]

    def __post_init__(self) -> None:
        """segment列がtimeline全体を一度だけ覆うことを検証する。"""
        if self.time_base <= 0 or not self.segments:
            msg = "Video Timelineにはtime baseとsegmentが必要です"
            raise ValueError(msg)
        expected_start = Fraction(0)
        for segment in self.segments:
            if segment.start != expected_start:
                msg = "Timeline Segmentにgapまたはoverlapがあります"
                raise ValueError(msg)
            expected_start = segment.end
        if expected_start != self.duration.seconds:
            msg = "Timeline SegmentがVideo Durationを覆っていません"
            raise ValueError(msg)

    def segment_at(self, video_time: Fraction) -> TimelineSegment:
        """Video Timeを所有する半開区間を返す。"""
        if video_time < 0 or video_time >= self.duration.seconds:
            msg = "Video TimeがVideo Durationの外側です"
            raise ValueError(msg)
        return next(
            segment
            for segment in self.segments
            if segment.start <= video_time < segment.end
        )
