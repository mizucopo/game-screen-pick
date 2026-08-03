"""Neutral Image Analysisから決定的なAnnotation shortlistを構築する。"""

import math
from dataclasses import replace
from fractions import Fraction

from ..models.candidate_annotation_request import CandidateAnnotationRequest
from ..models.candidate_moment import CandidateMoment
from ..models.context_cue import ContextCue
from ..models.frame_candidate import FrameCandidate
from ..models.video_stage_result import VideoStageResult

_CONTEXT_RADIUS_SECONDS = Fraction(15)
_CONTEXT_CUE_LIMIT = 3
_CUE_SELECTION_POLICY_VERSION = "nearby-context-v1"
_SCENE_CATALOG_REPRESENTATIVE_LIMIT = 24
_ANNOTATION_DIVERSITY_THRESHOLD = 0.72

type AnnotationShortlistItem = tuple[
    int,
    VideoStageResult,
    CandidateMoment,
    tuple[FrameCandidate, ...],
    FrameCandidate,
    Fraction,
]


def build_candidate_annotation_requests(
    video_stage_results: tuple[VideoStageResult, ...],
    *,
    selection_intent: str,
) -> tuple[CandidateAnnotationRequest, ...]:
    """各MomentのPrimaryと代替候補を保持してvisual diversity順に並べる。"""
    if not selection_intent.strip():
        msg = "Selection shortlistの意図が不正です"
        raise ValueError(msg)
    total_duration = sum(
        (result.scan.timeline.duration.seconds for result in video_stage_results),
        start=Fraction(0),
    )
    if total_duration <= 0:
        return ()

    candidates: list[AnnotationShortlistItem] = []
    elapsed = Fraction(0)
    for video_order, result in enumerate(video_stage_results, start=1):
        duration = result.scan.timeline.duration.seconds
        frames_by_id = {
            frame.identifier: frame for frame in result.extraction.candidates
        }
        for moment in result.extraction.moments:
            frames = tuple(
                frame
                for frame_id in moment.frame_candidate_ids
                if (frame := frames_by_id.get(frame_id)) is not None
                and frame.analysis is not None
                and frame.analysis.eligible
            )
            if not frames:
                continue
            if not 0 <= moment.anchor_time < duration:
                msg = "Candidate MomentがVideo Durationの範囲外です"
                raise ValueError(msg)
            ordered_frames = _ordered_local_frames(frames)
            representative = ordered_frames[0]
            normalized_moment = replace(
                moment,
                frame_candidate_ids=tuple(frame.identifier for frame in ordered_frames),
            )
            progress = (elapsed + moment.anchor_time) / total_duration
            candidates.append(
                (
                    video_order,
                    result,
                    normalized_moment,
                    ordered_frames,
                    representative,
                    progress,
                )
            )
        elapsed += duration

    base_order = sorted(candidates, key=_shortlist_base_key)
    ordered = _reserve_unique_fallback_frames(
        _unique_representative_frames(
            _diverse_prefix(base_order, _ANNOTATION_DIVERSITY_THRESHOLD)
        )
    )
    return tuple(
        CandidateAnnotationRequest(
            moment=moment,
            frame_candidates=frames,
            context_cues=_nearby_context_cues(
                result.context.annotation_cues,
                moment.anchor_time,
            ),
            video_set_progress=progress,
            selection_intent=selection_intent,
            cue_selection_policy_version=_CUE_SELECTION_POLICY_VERSION,
        )
        for _, result, moment, frames, _, progress in ordered
    )


def select_scene_catalog_representatives(
    requests: tuple[CandidateAnnotationRequest, ...],
    *,
    limit: int = _SCENE_CATALOG_REPRESENTATIVE_LIMIT,
) -> tuple[FrameCandidate, ...]:
    """shortlist全体から一意なlocal代表を要求枚数非依存で最大24枚返す。"""
    if limit < 1:
        msg = "Scene Catalog Representative上限は正の整数が必要です"
        raise ValueError(msg)
    representatives: list[FrameCandidate] = []
    seen_identifiers: set[str] = set()
    for request in requests:
        representative = _local_representative(request.frame_candidates)
        if representative.identifier in seen_identifiers:
            continue
        representatives.append(representative)
        seen_identifiers.add(representative.identifier)
        if len(representatives) == limit:
            break
    return tuple(representatives)


def _local_representative(
    frames: tuple[FrameCandidate, ...],
) -> FrameCandidate:
    """最高Quality Scoreを選び同点をFrame Candidate IDで固定する。"""
    return _ordered_local_frames(frames)[0]


def _ordered_local_frames(
    frames: tuple[FrameCandidate, ...],
) -> tuple[FrameCandidate, ...]:
    """Primaryを先頭に同一Momentのfallback候補を最大3件返す。"""
    return tuple(sorted(frames, key=_representative_key)[:3])


def _representative_key(frame: FrameCandidate) -> tuple[float, str]:
    """local代表の決定的な比較keyを返す。"""
    analysis = frame.analysis
    if analysis is None or not analysis.eligible:
        raise ValueError("local代表には適格なNeutral Image Analysisが必要です")
    return (-analysis.quality_score, frame.identifier)


def _shortlist_base_key(
    item: AnnotationShortlistItem,
) -> tuple[float, int, Fraction, str]:
    """Qualityを基礎に残る同点をdomain順で固定する。"""
    video_order, _, moment, _, representative, _ = item
    analysis = representative.analysis
    if analysis is None:
        raise AssertionError
    return (
        -analysis.quality_score,
        video_order,
        moment.anchor_time,
        moment.identifier,
    )


def _diverse_prefix(
    base_order: list[AnnotationShortlistItem],
    similarity_threshold: float,
) -> tuple[AnnotationShortlistItem, ...]:
    """base順を走査し既選抜と非類似のMomentを一つのprefixへ集める。"""
    diverse: list[AnnotationShortlistItem] = []
    deferred: list[AnnotationShortlistItem] = []
    for item in base_order:
        representative = item[4]
        if all(
            _cosine_similarity(representative, selected[4]) <= similarity_threshold
            for selected in diverse
        ):
            diverse.append(item)
        else:
            deferred.append(item)
    return (*diverse, *deferred)


def _unique_representative_frames(
    items: tuple[AnnotationShortlistItem, ...],
) -> tuple[AnnotationShortlistItem, ...]:
    """shortlist順で各Representative Frameを一つのMomentへ限定する。"""
    result: list[AnnotationShortlistItem] = []
    seen_identifiers: set[str] = set()
    for item in items:
        identifier = item[4].identifier
        if identifier in seen_identifiers:
            continue
        result.append(item)
        seen_identifiers.add(identifier)
    return tuple(result)


def _reserve_unique_fallback_frames(
    items: tuple[AnnotationShortlistItem, ...],
) -> tuple[AnnotationShortlistItem, ...]:
    """全Primaryを優先しfallback frameをshortlist全体で一意にする。"""
    reserved_identifiers = {item[4].identifier for item in items}
    result: list[AnnotationShortlistItem] = []
    for video_order, stage, moment, frames, primary, progress in items:
        unique_frames = [primary]
        for frame in frames[1:]:
            if frame.identifier in reserved_identifiers:
                continue
            unique_frames.append(frame)
            reserved_identifiers.add(frame.identifier)
        normalized_frames = tuple(unique_frames)
        result.append(
            (
                video_order,
                stage,
                replace(
                    moment,
                    frame_candidate_ids=tuple(
                        frame.identifier for frame in normalized_frames
                    ),
                ),
                normalized_frames,
                primary,
                progress,
            )
        )
    return tuple(result)


def _cosine_similarity(left: FrameCandidate, right: FrameCandidate) -> float:
    """Neutral Image Analysisのvisual feature cosine similarityを返す。"""
    left_analysis = left.analysis
    right_analysis = right.analysis
    if left_analysis is None or right_analysis is None:
        raise AssertionError
    left_feature = left_analysis.visual_feature
    right_feature = right_analysis.visual_feature
    if not left_feature or len(left_feature) != len(right_feature):
        msg = "Visual featureの次元が一致しません"
        raise ValueError(msg)
    numerator = sum(
        left_value * right_value
        for left_value, right_value in zip(left_feature, right_feature, strict=True)
    )
    left_norm = math.sqrt(sum(value * value for value in left_feature))
    right_norm = math.sqrt(sum(value * value for value in right_feature))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return max(-1.0, min(1.0, numerator / (left_norm * right_norm)))


def _nearby_context_cues(
    cues: tuple[ContextCue, ...],
    anchor: Fraction,
) -> tuple[ContextCue, ...]:
    """anchor前後15秒と重なるCueを距離選抜して時系列へ戻す。"""
    window_start = max(Fraction(0), anchor - _CONTEXT_RADIUS_SECONDS)
    window_end = anchor + _CONTEXT_RADIUS_SECONDS
    overlapping = tuple(
        cue for cue in cues if cue.end >= window_start and cue.start <= window_end
    )
    nearest = sorted(
        overlapping,
        key=lambda cue: (
            _interval_distance(cue, anchor),
            cue.start,
            cue.end,
            cue.identifier,
        ),
    )[:_CONTEXT_CUE_LIMIT]
    return tuple(sorted(nearest, key=lambda cue: (cue.start, cue.end, cue.identifier)))


def _interval_distance(cue: ContextCue, anchor: Fraction) -> Fraction:
    """Cue区間とanchor点の最短距離を返す。"""
    if cue.end < anchor:
        return anchor - cue.end
    if cue.start > anchor:
        return cue.start - anchor
    return Fraction(0)
