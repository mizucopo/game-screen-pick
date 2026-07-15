"""Candidate Moment周辺のnative frameを選抜する。"""

import math
from collections.abc import Iterable, Iterator
from dataclasses import replace
from fractions import Fraction

import numpy as np

from ..models.candidate_moment import CandidateMoment
from ..models.content_reject_reason import ContentRejectReason
from ..models.decoded_video_frame import DecodedVideoFrame
from ..models.frame_candidate import FrameCandidate
from ..models.frame_candidate_extraction import FrameCandidateExtraction
from ..models.neutral_image_analysis import NeutralImageAnalysis
from ..models.video_timeline import VideoTimeline
from .analyze_neutral_images import analyze_neutral_images
from .build_refinement_pts_ranges import build_refinement_pts_ranges
from .build_video_entity_id import build_video_entity_id

_FRAME_ID_ALGORITHM = "frame-candidate-id-v1"
_PERCEPTUAL_MAD_THRESHOLD = 2.0


def refine_candidate_moments(
    *,
    video_fingerprint: str,
    timeline: VideoTimeline,
    moments: tuple[CandidateMoment, ...],
    frames: tuple[DecodedVideoFrame, ...],
    refinement_radius_seconds: float,
    max_frame_candidates: int,
) -> FrameCandidateExtraction:
    """無効frame除外、Moment内dedupe、多様性選抜を適用する。"""
    groups = tuple(
        iter_refined_candidate_groups(
            video_fingerprint=video_fingerprint,
            timeline=timeline,
            moments=moments,
            frames=frames,
            refinement_radius_seconds=refinement_radius_seconds,
            max_frame_candidates=max_frame_candidates,
        )
    )
    return combine_refined_candidate_groups(moments, groups)


def iter_refined_candidate_groups(
    *,
    video_fingerprint: str,
    timeline: VideoTimeline,
    moments: tuple[CandidateMoment, ...],
    frames: Iterable[DecodedVideoFrame],
    refinement_radius_seconds: float,
    max_frame_candidates: int,
) -> Iterator[FrameCandidateExtraction]:
    """一回のframe streamを連続Refinement Window Groupごとに解析する。"""
    _validate_refinement_arguments(
        refinement_radius_seconds,
        max_frame_candidates,
    )
    pts_ranges = build_refinement_pts_ranges(
        timeline,
        moments,
        refinement_radius_seconds,
    )
    frame_iterator = iter(frames)
    pending: DecodedVideoFrame | None = None
    radius = Fraction(str(refinement_radius_seconds))
    for start_pts, end_pts in pts_ranges:
        group_frames: list[DecodedVideoFrame] = []
        while True:
            if pending is None:
                try:
                    frame = next(frame_iterator)
                except StopIteration:
                    break
            else:
                frame = pending
                pending = None
            if frame.pts < start_pts:
                msg = "refinement frameが要求PTS range外です"
                raise ValueError(msg)
            if frame.pts >= end_pts:
                pending = frame
                break
            group_frames.append(frame)
        group_moments = tuple(
            moment for moment in moments if start_pts <= moment.source_pts < end_pts
        )
        yield _refine_candidate_group(
            video_fingerprint=video_fingerprint,
            timeline=timeline,
            moments=group_moments,
            frames=tuple(group_frames),
            radius=radius,
            max_frame_candidates=max_frame_candidates,
        )
    if pending is not None:
        msg = "refinement frameが要求PTS range外です"
        raise ValueError(msg)
    try:
        next(frame_iterator)
    except StopIteration:
        return
    msg = "refinement frameが要求PTS range外です"
    raise ValueError(msg)


def combine_refined_candidate_groups(
    moments: tuple[CandidateMoment, ...],
    groups: tuple[FrameCandidateExtraction, ...],
) -> FrameCandidateExtraction:
    """順次確定したgroupを一つのVideo Source抽出結果へ統合する。"""
    refined_by_id = {
        moment.identifier: moment for group in groups for moment in group.moments
    }
    refined_moments = tuple(
        refined_by_id.get(moment.identifier, replace(moment, frame_candidate_ids=()))
        for moment in moments
    )
    candidate_by_id = {
        candidate.identifier: candidate
        for group in groups
        for candidate in group.candidates
    }
    candidates = tuple(
        sorted(
            candidate_by_id.values(),
            key=lambda candidate: (
                Fraction(0) if candidate.video_time is None else candidate.video_time
            ),
        )
    )
    reject_breakdown = ContentRejectReason.empty_breakdown()
    for group in groups:
        for reason, count in group.reject_breakdown.items():
            reject_breakdown[reason] += count
    return FrameCandidateExtraction(
        moments=refined_moments,
        candidates=candidates,
        native_frame_count=sum(group.native_frame_count for group in groups),
        reject_breakdown=reject_breakdown,
        deduplicated_frame_count=sum(
            group.deduplicated_frame_count for group in groups
        ),
        zero_frame_moment_count=sum(
            not moment.frame_candidate_ids for moment in refined_moments
        ),
    )


def _validate_refinement_arguments(
    refinement_radius_seconds: float,
    max_frame_candidates: int,
) -> None:
    if not math.isfinite(refinement_radius_seconds) or refinement_radius_seconds < 0:
        msg = "Frame Refinement半径は0以上の有限値である必要があります"
        raise ValueError(msg)
    if not 1 <= max_frame_candidates <= 3:
        msg = "最大Frame Candidate数は1以上3以下である必要があります"
        raise ValueError(msg)


def _refine_candidate_group(
    *,
    video_fingerprint: str,
    timeline: VideoTimeline,
    moments: tuple[CandidateMoment, ...],
    frames: tuple[DecodedVideoFrame, ...],
    radius: Fraction,
    max_frame_candidates: int,
) -> FrameCandidateExtraction:
    """一つの連続Refinement Window Groupを解析して選抜する。"""
    unique_frames = {
        frame.pts: frame
        for frame in frames
        if _is_in_any_refinement_window(frame, timeline, moments, radius)
    }
    ordered_frames = tuple(unique_frames[pts] for pts in sorted(unique_frames))
    analyses = analyze_neutral_images(ordered_frames)
    analysis_by_pts = {item.source_pts: item for item in analyses}
    frame_by_pts = {item.pts: item for item in ordered_frames}
    reject_breakdown = ContentRejectReason.empty_breakdown()
    for analysis in analyses:
        if analysis.reject_reason is not None:
            reject_breakdown[analysis.reject_reason.value] += 1

    candidate_by_pts: dict[int, FrameCandidate] = {}
    refined_moments: list[CandidateMoment] = []
    deduplicated_frame_count = 0
    for moment in moments:
        eligible = [
            (frame_by_pts[pts], analysis)
            for pts, analysis in analysis_by_pts.items()
            if analysis.eligible
            and _is_in_moment_window(
                _video_time(frame_by_pts[pts], timeline),
                moment.anchor_time,
                timeline.duration.seconds,
                radius,
            )
        ]
        deduplicated, removed_count = _deduplicate_for_moment(
            eligible,
            moment,
            timeline,
        )
        deduplicated_frame_count += removed_count
        selected = _select_diverse_frames(
            deduplicated,
            moment,
            timeline,
            max_frame_candidates,
        )
        candidate_ids: list[str] = []
        for frame, analysis in selected:
            candidate = candidate_by_pts.get(frame.pts)
            if candidate is None:
                video_time = _video_time(frame, timeline)
                candidate = FrameCandidate(
                    identifier=build_video_entity_id(
                        "frm",
                        _FRAME_ID_ALGORITHM,
                        video_fingerprint,
                        video_time,
                    ),
                    image_bytes=b"",
                    video_fingerprint=video_fingerprint,
                    stream_index=frame.stream_index,
                    source_pts=frame.pts,
                    origin_pts=timeline.origin_pts,
                    time_base=frame.time_base,
                    video_time=video_time,
                    analysis=analysis,
                    decoded_frame=frame,
                )
                candidate_by_pts[frame.pts] = candidate
            candidate_ids.append(candidate.identifier)
        refined_moments.append(
            replace(moment, frame_candidate_ids=tuple(candidate_ids))
        )

    candidates = tuple(
        sorted(
            candidate_by_pts.values(),
            key=lambda candidate: (
                Fraction(0) if candidate.video_time is None else candidate.video_time
            ),
        )
    )
    return FrameCandidateExtraction(
        moments=tuple(refined_moments),
        candidates=candidates,
        native_frame_count=len(ordered_frames),
        reject_breakdown=reject_breakdown,
        deduplicated_frame_count=deduplicated_frame_count,
        zero_frame_moment_count=sum(
            not moment.frame_candidate_ids for moment in refined_moments
        ),
    )


def _is_in_any_refinement_window(
    frame: DecodedVideoFrame,
    timeline: VideoTimeline,
    moments: tuple[CandidateMoment, ...],
    radius: Fraction,
) -> bool:
    video_time = _video_time(frame, timeline)
    return 0 <= video_time < timeline.duration.seconds and any(
        _is_in_moment_window(
            video_time,
            moment.anchor_time,
            timeline.duration.seconds,
            radius,
        )
        for moment in moments
    )


def _is_in_moment_window(
    video_time: Fraction,
    anchor_time: Fraction,
    duration: Fraction,
    radius: Fraction,
) -> bool:
    if radius == 0:
        return video_time == anchor_time
    start = max(Fraction(0), anchor_time - radius)
    end = min(duration, anchor_time + radius)
    return start <= video_time < end


def _video_time(
    frame: DecodedVideoFrame,
    timeline: VideoTimeline,
) -> Fraction:
    return Fraction(frame.pts - timeline.origin_pts) * frame.time_base


def _deduplicate_for_moment(
    eligible: list[tuple[DecodedVideoFrame, NeutralImageAnalysis]],
    moment: CandidateMoment,
    timeline: VideoTimeline,
) -> tuple[list[tuple[DecodedVideoFrame, NeutralImageAnalysis]], int]:
    ordered = sorted(
        eligible,
        key=lambda item: (
            -item[1].quality_score,
            abs(_video_time(item[0], timeline) - moment.anchor_time),
            _video_time(item[0], timeline),
        ),
    )
    kept: list[tuple[DecodedVideoFrame, NeutralImageAnalysis]] = []
    removed_count = 0
    for item in ordered:
        if any(
            _signature_mad(item[1], existing[1]) <= _PERCEPTUAL_MAD_THRESHOLD
            for existing in kept
        ):
            removed_count += 1
            continue
        kept.append(item)
    return kept, removed_count


def _select_diverse_frames(
    frames: list[tuple[DecodedVideoFrame, NeutralImageAnalysis]],
    moment: CandidateMoment,
    timeline: VideoTimeline,
    maximum: int,
) -> list[tuple[DecodedVideoFrame, NeutralImageAnalysis]]:
    if not frames:
        return []
    remaining = list(frames)
    selected = [remaining.pop(0)]
    while remaining and len(selected) < maximum:
        chosen = max(
            remaining,
            key=lambda item: (
                min(_visual_distance(item[1], existing[1]) for existing in selected),
                item[1].quality_score,
                -abs(_video_time(item[0], timeline) - moment.anchor_time),
                -_video_time(item[0], timeline),
            ),
        )
        selected.append(chosen)
        remaining.remove(chosen)
    return sorted(selected, key=lambda item: _video_time(item[0], timeline))


def _signature_mad(
    left: NeutralImageAnalysis,
    right: NeutralImageAnalysis,
) -> float:
    left_values = np.frombuffer(left.grayscale_signature, dtype=np.uint8).astype(
        np.int16
    )
    right_values = np.frombuffer(right.grayscale_signature, dtype=np.uint8).astype(
        np.int16
    )
    return float(np.mean(np.abs(left_values - right_values)))


def _visual_distance(
    left: NeutralImageAnalysis,
    right: NeutralImageAnalysis,
) -> float:
    return float(
        np.linalg.norm(
            np.asarray(left.visual_feature) - np.asarray(right.visual_feature)
        )
    )
