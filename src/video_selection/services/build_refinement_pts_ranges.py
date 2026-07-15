"""Candidate Moment windowをsource PTS rangeへ変換する。"""

import math
from fractions import Fraction

from ..models.candidate_moment import CandidateMoment
from ..models.video_timeline import VideoTimeline


def build_refinement_pts_ranges(
    timeline: VideoTimeline,
    moments: tuple[CandidateMoment, ...],
    radius_seconds: float,
) -> tuple[tuple[int, int], ...]:
    """clamp済み半開Video Time windowをmerge済みPTS rangeで返す。"""
    if not math.isfinite(radius_seconds) or radius_seconds < 0:
        msg = "Frame Refinement半径は0以上の有限値である必要があります"
        raise ValueError(msg)
    radius = Fraction(str(radius_seconds))
    ranges: list[tuple[int, int]] = []
    for moment in moments:
        if radius == 0:
            ranges.append((moment.source_pts, moment.source_pts + 1))
            continue
        start_time = max(Fraction(0), moment.anchor_time - radius)
        end_time = min(timeline.duration.seconds, moment.anchor_time + radius)
        start_pts = timeline.origin_pts + _ceil_fraction(
            start_time / timeline.time_base
        )
        end_pts = timeline.origin_pts + _ceil_fraction(end_time / timeline.time_base)
        if start_pts < end_pts:
            ranges.append((start_pts, end_pts))
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
            continue
        previous_start, previous_end = merged[-1]
        merged[-1] = (previous_start, max(previous_end, end))
    return tuple(merged)


def _ceil_fraction(value: Fraction) -> int:
    return -(-value.numerator // value.denominator)
