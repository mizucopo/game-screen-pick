"""注釈済みBlog Candidateから画像を決定的に選ぶ。"""

import hashlib
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import replace
from fractions import Fraction
from typing import Literal

from ..models.blog_candidate import BlogCandidate
from ..models.candidate_annotation import (
    SELECTION_COVERAGE_FACETS,
    SelectionCoverageFacet,
)
from ..models.rejected_blog_candidate import RejectedBlogCandidate
from ..models.selected_blog_image import SelectedBlogImage
from ..models.selection_rejection_reason import SelectionRejectionReason
from ..models.selection_score import SelectionScore
from ..models.video_set_selection_result import (
    CONDITIONAL_COVERAGE_MINIMUM_REQUEST_COUNT,
    VideoSetSelectionResult,
)

SpoilerSensitivity = Literal["low", "medium", "high"]

_EXPLANATION_VALUES = {
    "none": 0.0,
    "low": 1 / 3,
    "medium": 2 / 3,
    "high": 1.0,
}
_CONTEXT_VALUES = {
    "unavailable": 0.0,
    "none": 0.0,
    "weak": 0.5,
    "strong": 1.0,
}
_SPOILER_PENALTIES: Mapping[SpoilerSensitivity, Mapping[str, float]] = {
    "low": {"none": 0.0, "low": 0.0, "medium": 0.02, "high": 0.05},
    "medium": {"none": 0.0, "low": 0.01, "medium": 0.04, "high": 0.10},
    "high": {"none": 0.0, "low": 0.02, "medium": 0.08, "high": 0.18},
}
_SPOILER_SENSITIVITY_ORDER: tuple[SpoilerSensitivity, ...] = (
    "low",
    "medium",
    "high",
)
_COVERAGE_TYPES = ("normal_gameplay", "event", "menu")
_COVERAGE_PERCENTAGES = (70, 25, 5)
_SIMILARITY_RELAXATION_DELTAS = (0.03, 0.06, 0.10, 0.15)
_MAX_AUTOMATIC_SIMILARITY_CEILING = 0.97
_MAX_CONFIGURED_SIMILARITY_THRESHOLD = 0.98
_VISUAL_NEAR_DUPLICATE_THRESHOLD = 0.995
_VARIANT_GROUP_SIMILARITY_THRESHOLD = 0.95
_SIMILARITY_REJECTION_REASONS = frozenset(
    {
        SelectionRejectionReason.VISUAL_NEAR_DUPLICATE,
        SelectionRejectionReason.SIMILARITY_CEILING,
    }
)


def select_from_shortlist_batches(
    batches: Iterable[tuple[BlogCandidate, ...]],
    *,
    requested_count: int,
    spoiler_sensitivity: SpoilerSensitivity,
    similarity_threshold: float,
) -> VideoSetSelectionResult:
    """注釈済みShortlist batchを不足時だけ追加して全体を再選定する。"""
    accumulated: list[BlogCandidate] = []
    last_result: VideoSetSelectionResult | None = None
    expansion_count = 0
    for batch_index, batch in enumerate(batches):
        if not batch:
            msg = "Selection Shortlist expansion batchは空にできません"
            raise ValueError(msg)
        accumulated.extend(batch)
        last_result = select_video_set_images(
            tuple(accumulated),
            requested_count=requested_count,
            spoiler_sensitivity=spoiler_sensitivity,
            similarity_threshold=similarity_threshold,
        )
        expansion_count = batch_index
        if not last_result.shortfall:
            return replace(
                last_result,
                shortlist_expansion_count=expansion_count,
                all_candidate_moments_exhausted=False,
            )
    if last_result is None:
        last_result = select_video_set_images(
            (),
            requested_count=requested_count,
            spoiler_sensitivity=spoiler_sensitivity,
            similarity_threshold=similarity_threshold,
        )
    return replace(
        last_result,
        shortlist_expansion_count=expansion_count,
        all_candidate_moments_exhausted=True,
    )


def select_video_set_images(
    candidates: tuple[BlogCandidate, ...],
    *,
    requested_count: int,
    spoiler_sensitivity: SpoilerSensitivity,
    similarity_threshold: float,
) -> VideoSetSelectionResult:
    """Blog Candidate集合からMarginal Selection Utility順に選ぶ。"""
    _validate_inputs(
        candidates,
        requested_count,
        spoiler_sensitivity,
        similarity_threshold,
    )
    major_spoiler_limit: int | None = None
    for current_sensitivity in _SPOILER_SENSITIVITY_ORDER:
        result = _select_with_major_spoiler_limit(
            candidates,
            requested_count=requested_count,
            spoiler_sensitivity=current_sensitivity,
            similarity_threshold=similarity_threshold,
            major_spoiler_limit=major_spoiler_limit,
        )
        if current_sensitivity == spoiler_sensitivity:
            return result
        major_spoiler_limit = result.major_spoiler_selected_count
    raise AssertionError("検証済みSpoiler Sensitivityを選定できません")


def _select_with_major_spoiler_limit(
    candidates: tuple[BlogCandidate, ...],
    *,
    requested_count: int,
    spoiler_sensitivity: SpoilerSensitivity,
    similarity_threshold: float,
    major_spoiler_limit: int | None,
) -> VideoSetSelectionResult:
    """一つの感度とMajor Spoiler上限でgreedy選定する。"""
    targets = _coverage_targets(requested_count)
    actuals = dict.fromkeys((*_COVERAGE_TYPES, "title", "other"), 0)
    conditional_minimums = _conditional_coverage_minimums(
        candidates,
        requested_count,
    )
    conditional_actuals = dict.fromkeys(SELECTION_COVERAGE_FACETS, 0)
    variant_groups = _assign_variant_groups(candidates)
    selected: list[SelectedBlogImage] = []
    remaining = list(candidates)
    counterfactual_scores: dict[str, SelectionScore] = {}
    final_similarity_ceiling = similarity_threshold
    similarity_passes = _similarity_passes(similarity_threshold)
    similarity_pass_index = 0
    while similarity_pass_index < len(similarity_passes):
        similarity_pass = similarity_passes[similarity_pass_index]
        is_terminal_similarity_pass = (
            similarity_pass_index == len(similarity_passes) - 1
        )
        final_similarity_ceiling = similarity_pass
        restart_unrestricted_selection = False
        while remaining and len(selected) < requested_count:
            scored = [
                (
                    candidate,
                    _score(
                        candidate,
                        selected,
                        targets,
                        actuals,
                        requested_count,
                        spoiler_sensitivity,
                        similarity_pass,
                    ),
                )
                for candidate in remaining
            ]
            for candidate, score in scored:
                previous = counterfactual_scores.get(candidate.identifier)
                if (
                    previous is None
                    or score.marginal_utility > previous.marginal_utility
                ):
                    counterfactual_scores[candidate.identifier] = score
            evaluated = [
                (candidate, score)
                for candidate, score in scored
                if _has_explanation_value(candidate)
                and _is_visually_eligible(
                    candidate,
                    remaining,
                    selected,
                    actuals,
                    similarity_pass,
                    variant_groups,
                    major_spoiler_limit,
                )
                and not (
                    candidate.annotation.blog_image_type == "title"
                    and actuals["title"] >= 1
                )
                and not _major_spoiler_limit_reached(
                    candidate,
                    selected,
                    major_spoiler_limit,
                )
            ]
            if not evaluated:
                break
            unmet_facets = {
                facet
                for facet in SELECTION_COVERAGE_FACETS
                if conditional_actuals[facet] < conditional_minimums[facet]
            }
            required_coverage_candidates = [
                (candidate, score)
                for candidate, score in evaluated
                if candidate.annotation.selection_coverage_facet in unmet_facets
            ]
            if required_coverage_candidates:
                evaluated = required_coverage_candidates
            elif unmet_facets and _has_remaining_coverage_candidate(
                remaining,
                unmet_facets,
            ):
                if not is_terminal_similarity_pass:
                    break
                for facet in unmet_facets:
                    conditional_minimums[facet] = 0
                for remaining_candidate in remaining:
                    counterfactual_scores.pop(remaining_candidate.identifier, None)
                similarity_pass_index = 0
                restart_unrestricted_selection = True
                break
            candidate, score = min(
                evaluated,
                key=lambda item: _selection_key(item[0], item[1]),
            )
            selected_coverage_facet = candidate.annotation.selection_coverage_facet
            minimum_coverage_facet = (
                selected_coverage_facet
                if selected_coverage_facet in unmet_facets
                else None
            )
            tie_break_applied = any(
                other.identifier != candidate.identifier
                and math.isclose(
                    other_score.marginal_utility,
                    score.marginal_utility,
                    rel_tol=0,
                    abs_tol=1e-12,
                )
                for other, other_score in evaluated
            )
            selected.append(
                SelectedBlogImage(
                    candidate=candidate,
                    selection_index=len(selected) + 1,
                    score=score,
                    reason_codes=_reason_codes(
                        candidate,
                        score,
                        variant_groups[candidate.identifier]
                        in {
                            item.variant_group_id
                            for item in selected
                            if item.candidate.annotation.scene_slug
                            == candidate.annotation.scene_slug
                        },
                        tie_break_applied,
                        minimum_coverage_facet,
                    ),
                    variant_group_id=variant_groups[candidate.identifier],
                    tie_break_applied=tie_break_applied,
                )
            )
            actuals[candidate.annotation.blog_image_type] += 1
            if selected_coverage_facet is not None:
                conditional_actuals[selected_coverage_facet] += 1
            remaining.remove(candidate)
        if len(selected) >= requested_count:
            break
        if restart_unrestricted_selection:
            continue
        similarity_pass_index += 1
    rejected = [
        _rejection(
            candidate,
            selected,
            variant_groups[candidate.identifier],
            counterfactual_scores[candidate.identifier],
            major_spoiler_limit,
            final_similarity_ceiling,
        )
        for candidate in remaining
    ]
    rejected.sort(
        key=lambda item: _selection_key(
            item.candidate,
            item.counterfactual_score,
        )
    )
    return VideoSetSelectionResult(
        selected=tuple(selected),
        rejected=tuple(rejected),
        requested_count=requested_count,
        blog_image_type_targets=targets,
        blog_image_type_actuals=actuals,
        final_similarity_ceiling=final_similarity_ceiling,
        major_spoiler_limit=major_spoiler_limit,
        annotated_candidate_count=len(candidates),
        shortlist_expansion_count=0,
        all_candidate_moments_exhausted=True,
    )


def _similarity_passes(base_threshold: float) -> tuple[float, ...]:
    terminal_ceiling = max(base_threshold, _MAX_AUTOMATIC_SIMILARITY_CEILING)
    candidates = (
        base_threshold,
        *(
            min(base_threshold + delta, terminal_ceiling)
            for delta in _SIMILARITY_RELAXATION_DELTAS
        ),
        terminal_ceiling,
    )
    return tuple(dict.fromkeys(candidates))


def _validate_inputs(
    candidates: tuple[BlogCandidate, ...],
    requested_count: int,
    spoiler_sensitivity: str,
    similarity_threshold: float,
) -> None:
    identifiers = tuple(candidate.identifier for candidate in candidates)
    moment_ids = tuple(
        candidate.annotation.candidate_moment_id for candidate in candidates
    )
    shortlist_ranks = tuple(candidate.shortlist_rank for candidate in candidates)
    feature_lengths = {len(candidate.visual_feature) for candidate in candidates}
    if (
        requested_count < 1
        or spoiler_sensitivity not in _SPOILER_PENALTIES
        or not math.isfinite(similarity_threshold)
        or not 0 <= similarity_threshold <= _MAX_CONFIGURED_SIMILARITY_THRESHOLD
        or len(identifiers) != len(set(identifiers))
        or len(moment_ids) != len(set(moment_ids))
        or len(shortlist_ranks) != len(set(shortlist_ranks))
        or len(feature_lengths) > 1
    ):
        msg = "Video Set selectorの候補、要求枚数、設定が不正です"
        raise ValueError(msg)


def _coverage_targets(requested_count: int) -> dict[str, int]:
    floors = [
        requested_count * percentage // 100 for percentage in _COVERAGE_PERCENTAGES
    ]
    remaining = requested_count - sum(floors)
    remainders = [
        requested_count * percentage % 100 for percentage in _COVERAGE_PERCENTAGES
    ]
    allocation_order = sorted(
        range(len(_COVERAGE_TYPES)),
        key=lambda index: (-remainders[index], index),
    )
    for index in allocation_order[:remaining]:
        floors[index] += 1
    targets = {
        image_type: floors[index] for index, image_type in enumerate(_COVERAGE_TYPES)
    }
    targets.update({"title": 0, "other": 0})
    return targets


def _conditional_coverage_minimums(
    candidates: tuple[BlogCandidate, ...],
    requested_count: int,
) -> dict[SelectionCoverageFacet, int]:
    """要求10枚以上で有効候補があるfacetだけ最低1枚にする。"""
    minimums = dict.fromkeys(SELECTION_COVERAGE_FACETS, 0)
    if requested_count < CONDITIONAL_COVERAGE_MINIMUM_REQUEST_COUNT:
        return minimums
    for candidate in candidates:
        facet = candidate.annotation.selection_coverage_facet
        if facet is not None and _has_explanation_value(candidate):
            minimums[facet] = 1
    return minimums


def _has_remaining_coverage_candidate(
    remaining: list[BlogCandidate],
    unmet_facets: set[SelectionCoverageFacet],
) -> bool:
    """後続similarity passで再評価すべき最低coverage候補が残るかを返す。"""
    return any(
        candidate.annotation.selection_coverage_facet in unmet_facets
        and _has_explanation_value(candidate)
        for candidate in remaining
    )


def _score(
    candidate: BlogCandidate,
    selected: list[SelectedBlogImage],
    targets: Mapping[str, int],
    actuals: Mapping[str, int],
    requested_count: int,
    spoiler_sensitivity: SpoilerSensitivity,
    similarity_pass: float,
) -> SelectionScore:
    annotation = candidate.annotation
    base_utility = (
        0.70 * candidate.quality_score
        + 0.25 * _EXPLANATION_VALUES[annotation.explanation_value]
        + 0.05 * _CONTEXT_VALUES[annotation.context_relevance]
    )
    spoiler_penalty = _SPOILER_PENALTIES[spoiler_sensitivity][annotation.spoiler_risk]
    coverage_bonus = _coverage_bonus(candidate, targets, actuals)
    temporal_penalty = _temporal_penalty(candidate, selected, requested_count)
    nearest_similarity = _nearest_selected_similarity(candidate, selected)
    return SelectionScore(
        base_utility=base_utility,
        spoiler_penalty=spoiler_penalty,
        coverage_bonus=coverage_bonus,
        temporal_diversity_penalty=temporal_penalty,
        marginal_utility=(
            base_utility + coverage_bonus - spoiler_penalty - temporal_penalty
        ),
        similarity_pass=similarity_pass,
        nearest_selected_similarity=nearest_similarity,
    )


def _coverage_bonus(
    candidate: BlogCandidate,
    targets: Mapping[str, int],
    actuals: Mapping[str, int],
) -> float:
    image_type = candidate.annotation.blog_image_type
    if image_type in _COVERAGE_TYPES and actuals[image_type] < targets[image_type]:
        return 0.10
    if image_type == "title" and actuals["title"] == 0:
        return 0.05
    return 0.0


def _temporal_penalty(
    candidate: BlogCandidate,
    selected: list[SelectedBlogImage],
    requested_count: int,
) -> float:
    if not selected:
        return 0.0
    nearest_distance = min(
        abs(candidate.video_set_progress - item.candidate.video_set_progress)
        for item in selected
    )
    spacing = Fraction(1, requested_count)
    return 0.08 * max(0.0, 1.0 - float(nearest_distance / spacing))


def _is_visually_eligible(
    candidate: BlogCandidate,
    remaining: list[BlogCandidate],
    selected: list[SelectedBlogImage],
    actuals: Mapping[str, int],
    similarity_threshold: float,
    variant_groups: Mapping[str, str],
    major_spoiler_limit: int | None,
) -> bool:
    nearest = _nearest_selected_similarity(candidate, selected)
    if nearest is not None and nearest > similarity_threshold:
        return False
    candidate_group = variant_groups[candidate.identifier]
    selected_groups = {
        item.variant_group_id
        for item in selected
        if item.candidate.annotation.scene_slug == candidate.annotation.scene_slug
    }
    return not (
        candidate.scene_selection_role == "recurring_gameplay"
        and candidate_group in selected_groups
        and _has_unrepresented_eligible_variant_group(
            candidate,
            remaining,
            selected,
            actuals,
            similarity_threshold,
            variant_groups,
            selected_groups,
            major_spoiler_limit,
        )
    )


def _has_explanation_value(candidate: BlogCandidate) -> bool:
    return candidate.annotation.explanation_value != "none"


def _has_unrepresented_eligible_variant_group(
    candidate: BlogCandidate,
    remaining: list[BlogCandidate],
    selected: list[SelectedBlogImage],
    actuals: Mapping[str, int],
    similarity_threshold: float,
    variant_groups: Mapping[str, str],
    selected_groups: set[str],
    major_spoiler_limit: int | None,
) -> bool:
    for alternative in remaining:
        if (
            not _has_explanation_value(alternative)
            or alternative.annotation.scene_slug != candidate.annotation.scene_slug
            or variant_groups[alternative.identifier] in selected_groups
            or (
                alternative.annotation.blog_image_type == "title"
                and actuals["title"] >= 1
            )
            or _major_spoiler_limit_reached(
                alternative,
                selected,
                major_spoiler_limit,
            )
        ):
            continue
        nearest = _nearest_selected_similarity(alternative, selected)
        if nearest is None or nearest <= similarity_threshold:
            return True
    return False


def _nearest_selected_similarity(
    candidate: BlogCandidate,
    selected: list[SelectedBlogImage],
) -> float | None:
    if not selected:
        return None
    return max(_cosine_similarity(candidate, item.candidate) for item in selected)


def _cosine_similarity(left: BlogCandidate, right: BlogCandidate) -> float:
    numerator = sum(
        left_value * right_value
        for left_value, right_value in zip(
            left.visual_feature,
            right.visual_feature,
            strict=True,
        )
    )
    left_norm = math.sqrt(sum(value * value for value in left.visual_feature))
    right_norm = math.sqrt(sum(value * value for value in right.visual_feature))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return max(-1.0, min(1.0, numerator / (left_norm * right_norm)))


def _selection_key(
    candidate: BlogCandidate,
    score: SelectionScore,
) -> tuple[float, float, float, float, int, Fraction, str]:
    nearest_similarity = score.nearest_selected_similarity
    return (
        -score.marginal_utility,
        score.spoiler_penalty,
        -candidate.quality_score,
        -1.0 if nearest_similarity is None else nearest_similarity,
        candidate.video_order,
        candidate.annotation.candidate.video_time or Fraction(0),
        candidate.identifier,
    )


def _reason_codes(
    candidate: BlogCandidate,
    score: SelectionScore,
    is_variant_expansion: bool,
    tie_break_applied: bool,
    minimum_coverage_facet: SelectionCoverageFacet | None,
) -> tuple[str, ...]:
    annotation = candidate.annotation
    reasons: list[str] = []
    if candidate.quality_score >= 0.8:
        reasons.append("high_quality")
    if annotation.explanation_value == "high":
        reasons.append("high_explanation_value")
    if annotation.context_relevance == "strong":
        reasons.append("strong_context_relevance")
    if score.coverage_bonus > 0:
        reasons.append(
            "title_first_image_bonus"
            if annotation.blog_image_type == "title"
            else f"{annotation.blog_image_type}_coverage"
        )
    if minimum_coverage_facet is not None:
        reasons.append(f"{minimum_coverage_facet}_minimum_coverage")
    if is_variant_expansion and candidate.scene_selection_role == "recurring_gameplay":
        reasons.append("recurring_gameplay_variant")
    if score.spoiler_penalty > 0:
        reasons.append(f"{annotation.spoiler_risk}_spoiler_penalty_applied")
    if tie_break_applied:
        reasons.append("stable_tie_break")
    return tuple(reasons)


def _rejection(
    candidate: BlogCandidate,
    selected: list[SelectedBlogImage],
    variant_group_id: str,
    counterfactual_score: SelectionScore,
    major_spoiler_limit: int | None,
    final_similarity_ceiling: float,
) -> RejectedBlogCandidate:
    nearest_similarity = _nearest_selected_similarity(candidate, selected)
    nearest_selected = _nearest_selected(candidate, selected)
    selected_title = next(
        (
            item
            for item in selected
            if item.candidate.annotation.blog_image_type == "title"
        ),
        None,
    )
    if candidate.annotation.blog_image_type == "title" and selected_title is not None:
        reason = SelectionRejectionReason.TITLE_LIMIT
    elif (
        nearest_similarity is not None
        and nearest_similarity > _VISUAL_NEAR_DUPLICATE_THRESHOLD
    ):
        reason = SelectionRejectionReason.VISUAL_NEAR_DUPLICATE
    elif (
        nearest_similarity is not None and nearest_similarity > final_similarity_ceiling
    ):
        reason = SelectionRejectionReason.SIMILARITY_CEILING
    elif _major_spoiler_limit_reached(
        candidate,
        selected,
        major_spoiler_limit,
    ):
        reason = SelectionRejectionReason.SPOILER_MONOTONICITY_GUARD
    else:
        reason = SelectionRejectionReason.LOWER_MARGINAL_UTILITY
    return RejectedBlogCandidate(
        candidate=candidate,
        reason_code=reason,
        counterfactual_score=counterfactual_score,
        blocked_by_image_id=(
            selected_title.candidate.identifier
            if reason is SelectionRejectionReason.TITLE_LIMIT
            and selected_title is not None
            else None
        ),
        nearest_selected_image_id=(
            nearest_selected.candidate.identifier
            if reason in _SIMILARITY_REJECTION_REASONS and nearest_selected is not None
            else None
        ),
        similarity=(
            nearest_similarity if reason in _SIMILARITY_REJECTION_REASONS else None
        ),
        variant_group_id=variant_group_id,
    )


def _nearest_selected(
    candidate: BlogCandidate,
    selected: list[SelectedBlogImage],
) -> SelectedBlogImage | None:
    if not selected:
        return None
    return min(
        selected,
        key=lambda item: (
            -_cosine_similarity(candidate, item.candidate),
            item.candidate.identifier,
        ),
    )


def _assign_variant_groups(
    candidates: tuple[BlogCandidate, ...],
) -> dict[str, str]:
    by_scene: dict[str, list[BlogCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_scene[candidate.annotation.scene_slug].append(candidate)

    result: dict[str, str] = {}
    for scene_slug in sorted(by_scene):
        scene_candidates = sorted(
            by_scene[scene_slug],
            key=lambda item: item.identifier,
        )
        unassigned = {item.identifier for item in scene_candidates}
        by_identifier = {item.identifier: item for item in scene_candidates}
        while unassigned:
            root = min(unassigned)
            component = {root}
            frontier = [root]
            unassigned.remove(root)
            while frontier:
                current_id = frontier.pop()
                current = by_identifier[current_id]
                connected = {
                    other_id
                    for other_id in unassigned
                    if _cosine_similarity(current, by_identifier[other_id])
                    >= _VARIANT_GROUP_SIMILARITY_THRESHOLD
                }
                component.update(connected)
                frontier.extend(sorted(connected, reverse=True))
                unassigned.difference_update(connected)
            group_payload = "\0".join((scene_slug, *sorted(component)))
            group_id = "variant_" + hashlib.sha256(group_payload.encode()).hexdigest()
            result.update(dict.fromkeys(component, group_id))
    return result


def _major_spoiler_limit_reached(
    candidate: BlogCandidate,
    selected: list[SelectedBlogImage],
    major_spoiler_limit: int | None,
) -> bool:
    if candidate.annotation.spoiler_risk != "high" or major_spoiler_limit is None:
        return False
    selected_count = sum(
        item.candidate.annotation.spoiler_risk == "high" for item in selected
    )
    return selected_count >= major_spoiler_limit
