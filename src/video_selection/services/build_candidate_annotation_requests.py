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


def build_candidate_annotation_requests(
    video_stage_results: tuple[VideoStageResult, ...],
    *,
    selection_intent: str,
    similarity_threshold: float,
) -> tuple[CandidateAnnotationRequest, ...]:
    """各Momentを一つのlocal代表へ絞りvisual diversity順に並べる。"""
    if (
        not selection_intent.strip()
        or not math.isfinite(similarity_threshold)
        or not 0 <= similarity_threshold <= 1
    ):
        msg = "Selection shortlistの意図またはsimilarity thresholdが不正です"
        raise ValueError(msg)
    total_duration = sum(
        (result.scan.timeline.duration.seconds for result in video_stage_results),
        start=Fraction(0),
    )
    if total_duration <= 0:
        return ()

    candidates: list[
        tuple[
            int,
            VideoStageResult,
            CandidateMoment,
            tuple[FrameCandidate, ...],
            FrameCandidate,
            Fraction,
        ]
    ] = []
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
            representative = _local_representative(frames)
            normalized_moment = replace(
                moment,
                frame_candidate_ids=(representative.identifier,),
            )
            progress = (elapsed + moment.anchor_time) / total_duration
            candidates.append(
                (
                    video_order,
                    result,
                    normalized_moment,
                    (representative,),
                    representative,
                    progress,
                )
            )
        elapsed += duration

    base_order = sorted(candidates, key=_shortlist_base_key)
    ordered = _diverse_prefix(base_order, similarity_threshold)
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
    """shortlist全体から要求枚数非依存のlocal代表を最大24枚返す。"""
    if limit < 1:
        msg = "Scene Catalog Representative上限は正の整数が必要です"
        raise ValueError(msg)
    return tuple(
        _local_representative(request.frame_candidates) for request in requests[:limit]
    )


def _local_representative(
    frames: tuple[FrameCandidate, ...],
) -> FrameCandidate:
    """最高Quality Scoreを選び同点をFrame Candidate IDで固定する。"""
    return min(frames, key=_representative_key)


def _representative_key(frame: FrameCandidate) -> tuple[float, str]:
    """local代表の決定的な比較keyを返す。"""
    analysis = frame.analysis
    if analysis is None or not analysis.eligible:
        raise ValueError("local代表には適格なNeutral Image Analysisが必要です")
    return (-analysis.quality_score, frame.identifier)


def _shortlist_base_key(
    item: tuple[
        int,
        VideoStageResult,
        CandidateMoment,
        tuple[FrameCandidate, ...],
        FrameCandidate,
        Fraction,
    ],
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
    base_order: list[
        tuple[
            int,
            VideoStageResult,
            CandidateMoment,
            tuple[FrameCandidate, ...],
            FrameCandidate,
            Fraction,
        ]
    ],
    similarity_threshold: float,
) -> tuple[
    tuple[
        int,
        VideoStageResult,
        CandidateMoment,
        tuple[FrameCandidate, ...],
        FrameCandidate,
        Fraction,
    ],
    ...,
]:
    """base順を走査し既選抜と非類似のMomentを一つのprefixへ集める。"""
    diverse: list[
        tuple[
            int,
            VideoStageResult,
            CandidateMoment,
            tuple[FrameCandidate, ...],
            FrameCandidate,
            Fraction,
        ]
    ] = []
    deferred: list[
        tuple[
            int,
            VideoStageResult,
            CandidateMoment,
            tuple[FrameCandidate, ...],
            FrameCandidate,
            Fraction,
        ]
    ] = []
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
