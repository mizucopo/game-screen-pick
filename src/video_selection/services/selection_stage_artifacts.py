"""Video Set Selection Resultのcache artifact変換。"""

import math
from collections.abc import Mapping
from typing import cast

from ..models.blog_candidate import BlogCandidate
from ..models.combat_subject_evidence import (
    COMBAT_SUBJECT_BODY_PLANS,
    COMBAT_SUBJECT_COLORS,
    COMBAT_SUBJECT_SCALES,
    COMBAT_SUBJECT_SURFACES,
    COMBAT_SUBJECT_TRAITS,
)
from ..models.rejected_blog_candidate import RejectedBlogCandidate
from ..models.selected_blog_image import SelectedBlogImage
from ..models.selection_rejection_reason import SelectionRejectionReason
from ..models.selection_score import SelectionScore
from ..models.semantic_duplicate_basis import (
    SEMANTIC_DUPLICATE_BASES,
    SemanticDuplicateBasis,
)
from ..models.video_set_selection_result import VideoSetSelectionResult

_SELECTION_SCHEMA = "game-screen-pick/video-set-selection@3.0.0"


def serialize_video_set_selection_result(
    selection: VideoSetSelectionResult,
) -> dict[str, object]:
    """選定結果をcandidate IDへ結び付いたJSON artifactへ変換する。"""
    return {
        "schema": _SELECTION_SCHEMA,
        "selected": [
            {
                "candidate_id": item.candidate.identifier,
                "selection_index": item.selection_index,
                "score": _serialize_score(item.score),
                "reason_codes": list(item.reason_codes),
                "variant_group_id": item.variant_group_id,
                "tie_break_applied": item.tie_break_applied,
                "semantic_group_id": item.semantic_group_id,
                "semantic_group_basis": item.semantic_group_basis,
                "semantic_group_evidence": (
                    list(item.semantic_group_evidence)
                    if item.semantic_group_evidence is not None
                    else None
                ),
            }
            for item in selection.selected
        ],
        "rejected": [
            {
                "candidate_id": item.candidate.identifier,
                "reason_code": item.reason_code.value,
                "counterfactual_score": _serialize_score(item.counterfactual_score),
                "blocked_by_image_id": item.blocked_by_image_id,
                "nearest_selected_image_id": item.nearest_selected_image_id,
                "similarity": item.similarity,
                "variant_group_id": item.variant_group_id,
                "semantic_group_id": item.semantic_group_id,
                "semantic_group_basis": item.semantic_group_basis,
                "semantic_group_evidence": (
                    list(item.semantic_group_evidence)
                    if item.semantic_group_evidence is not None
                    else None
                ),
            }
            for item in selection.rejected
        ],
        "requested_count": selection.requested_count,
        "blog_image_type_targets": selection.blog_image_type_targets,
        "blog_image_type_actuals": selection.blog_image_type_actuals,
        "final_similarity_ceiling": selection.final_similarity_ceiling,
        "major_spoiler_limit": selection.major_spoiler_limit,
        "annotated_candidate_count": selection.annotated_candidate_count,
        "shortlist_expansion_count": selection.shortlist_expansion_count,
        "all_candidate_moments_exhausted": (selection.all_candidate_moments_exhausted),
    }


def selection_artifact_candidate_count(artifact: Mapping[str, object]) -> int:
    """cache復元前に必要な注釈済みcandidate件数を返す。"""
    if artifact.get("schema") != _SELECTION_SCHEMA:
        raise ValueError("Video Set Selection artifact schemaが不正です")
    return _integer(
        artifact.get("annotated_candidate_count"),
        "annotated_candidate_count",
    )


def restore_video_set_selection_result(
    artifact: Mapping[str, object],
    candidates: tuple[BlogCandidate, ...],
) -> VideoSetSelectionResult:
    """検証済みartifactを現在の注釈済みcandidateへ結び直す。"""
    annotated_candidate_count = selection_artifact_candidate_count(artifact)
    candidate_by_id = {item.identifier: item for item in candidates}
    if len(candidate_by_id) != len(candidates) or annotated_candidate_count != len(
        candidates
    ):
        raise ValueError("Video Set Selection candidate集合が不正です")
    raw_selected = artifact.get("selected")
    raw_rejected = artifact.get("rejected")
    if not isinstance(raw_selected, list) or not isinstance(raw_rejected, list):
        raise ValueError("Video Set Selection artifact itemが不正です")
    selected = tuple(
        _restore_selected(item, candidate_by_id, expected_index=index)
        for index, item in enumerate(raw_selected, start=1)
    )
    rejected = tuple(_restore_rejected(item, candidate_by_id) for item in raw_rejected)
    restored_ids = (
        *(item.candidate.identifier for item in selected),
        *(item.candidate.identifier for item in rejected),
    )
    if len(restored_ids) != len(set(restored_ids)) or set(restored_ids) != set(
        candidate_by_id
    ):
        raise ValueError("Video Set Selection candidate集合が不正です")
    requested_count = _integer(artifact.get("requested_count"), "requested_count")
    if requested_count < 1 or len(selected) > requested_count:
        raise ValueError("Video Set Selection requested countが不正です")
    final_similarity_ceiling = _number(
        artifact.get("final_similarity_ceiling"),
        "final_similarity_ceiling",
    )
    if not 0 <= final_similarity_ceiling <= 1:
        raise ValueError("Video Set Selection similarity ceilingが不正です")
    major_spoiler_limit_value = artifact.get("major_spoiler_limit")
    major_spoiler_limit = (
        None
        if major_spoiler_limit_value is None
        else _integer(major_spoiler_limit_value, "major_spoiler_limit")
    )
    exhausted = artifact.get("all_candidate_moments_exhausted")
    if not isinstance(exhausted, bool):
        raise ValueError("Video Set Selection exhausted flagが不正です")
    return VideoSetSelectionResult(
        selected=selected,
        rejected=rejected,
        requested_count=requested_count,
        blog_image_type_targets=_integer_mapping(
            artifact.get("blog_image_type_targets"),
            "blog_image_type_targets",
        ),
        blog_image_type_actuals=_integer_mapping(
            artifact.get("blog_image_type_actuals"),
            "blog_image_type_actuals",
        ),
        final_similarity_ceiling=final_similarity_ceiling,
        major_spoiler_limit=major_spoiler_limit,
        annotated_candidate_count=annotated_candidate_count,
        shortlist_expansion_count=_integer(
            artifact.get("shortlist_expansion_count"),
            "shortlist_expansion_count",
        ),
        all_candidate_moments_exhausted=exhausted,
    )


def _restore_selected(
    value: object,
    candidates: Mapping[str, BlogCandidate],
    *,
    expected_index: int,
) -> SelectedBlogImage:
    item = _mapping(value, "selected")
    candidate = _candidate(item.get("candidate_id"), candidates)
    selection_index = _integer(item.get("selection_index"), "selection_index")
    reason_codes = item.get("reason_codes")
    variant_group_id = item.get("variant_group_id")
    tie_break_applied = item.get("tie_break_applied")
    (
        semantic_group_id,
        semantic_group_basis,
        semantic_group_evidence,
    ) = _semantic_group(item)
    if (
        selection_index != expected_index
        or not isinstance(reason_codes, list)
        or not all(isinstance(reason, str) and reason for reason in reason_codes)
        or not isinstance(variant_group_id, str)
        or not variant_group_id
        or not isinstance(tie_break_applied, bool)
    ):
        raise ValueError("Video Set Selection selected itemが不正です")
    return SelectedBlogImage(
        candidate=candidate,
        selection_index=selection_index,
        score=_restore_score(item.get("score")),
        reason_codes=tuple(cast(list[str], reason_codes)),
        variant_group_id=variant_group_id,
        tie_break_applied=tie_break_applied,
        semantic_group_id=semantic_group_id,
        semantic_group_basis=semantic_group_basis,
        semantic_group_evidence=semantic_group_evidence,
    )


def _restore_rejected(
    value: object,
    candidates: Mapping[str, BlogCandidate],
) -> RejectedBlogCandidate:
    item = _mapping(value, "rejected")
    reason_code_value = item.get("reason_code")
    variant_group_id = item.get("variant_group_id")
    (
        semantic_group_id,
        semantic_group_basis,
        semantic_group_evidence,
    ) = _semantic_group(item)
    if not isinstance(reason_code_value, str):
        raise ValueError("Video Set Selection rejection reasonが不正です")
    try:
        reason_code = SelectionRejectionReason(reason_code_value)
    except ValueError:
        raise ValueError("Video Set Selection rejection reasonが不正です") from None
    if not isinstance(variant_group_id, str) or not variant_group_id:
        raise ValueError("Video Set Selection rejected itemが不正です")
    return RejectedBlogCandidate(
        candidate=_candidate(item.get("candidate_id"), candidates),
        reason_code=reason_code,
        counterfactual_score=_restore_score(item.get("counterfactual_score")),
        blocked_by_image_id=_optional_string(item.get("blocked_by_image_id")),
        nearest_selected_image_id=_optional_string(
            item.get("nearest_selected_image_id")
        ),
        similarity=_optional_number(item.get("similarity"), "similarity"),
        variant_group_id=variant_group_id,
        semantic_group_id=semantic_group_id,
        semantic_group_basis=semantic_group_basis,
        semantic_group_evidence=semantic_group_evidence,
    )


def _semantic_group(
    item: Mapping[str, object],
) -> tuple[
    str | None,
    SemanticDuplicateBasis | None,
    tuple[str, ...] | None,
]:
    """artifactのSemantic Duplicate Group fieldを検証して返す。"""
    group_id = item.get("semantic_group_id")
    basis = item.get("semantic_group_basis")
    raw_evidence = item.get("semantic_group_evidence")
    if group_id is None and basis is None and raw_evidence is None:
        return None, None, None
    if (
        not isinstance(group_id, str)
        or not group_id.startswith("semantic_")
        or len(group_id) != 73
        or any(character not in "0123456789abcdef" for character in group_id[9:])
        or basis not in SEMANTIC_DUPLICATE_BASES
    ):
        raise ValueError("Video Set Selection Semantic Duplicate Groupが不正です")
    evidence = _semantic_group_evidence(raw_evidence, basis)
    return group_id, basis, evidence


def _semantic_group_evidence(
    value: object,
    basis: SemanticDuplicateBasis,
) -> tuple[str, ...] | None:
    """Combat Subject Groupの公開可能な有限enum tokenだけを受理する。"""
    if basis != "combat_subject_appearance":
        if value is not None:
            raise ValueError(
                "Video Set Selection Semantic Duplicate evidenceが不正です"
            )
        return None
    allowed = {
        *(f"body_plan:{item}" for item in COMBAT_SUBJECT_BODY_PLANS),
        *(f"scale:{item}" for item in COMBAT_SUBJECT_SCALES),
        *(f"surface:{item}" for item in COMBAT_SUBJECT_SURFACES),
        *(f"color:{item}" for item in COMBAT_SUBJECT_COLORS),
        *(f"trait:{item}" for item in COMBAT_SUBJECT_TRAITS),
    }
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item in allowed for item in value)
        or len(value) != len(set(value))
    ):
        raise ValueError("Video Set Selection Semantic Duplicate evidenceが不正です")
    return tuple(cast(list[str], value))


def _serialize_score(score: SelectionScore) -> dict[str, float | None]:
    return {
        "base_utility": score.base_utility,
        "spoiler_penalty": score.spoiler_penalty,
        "coverage_bonus": score.coverage_bonus,
        "temporal_diversity_penalty": score.temporal_diversity_penalty,
        "marginal_utility": score.marginal_utility,
        "similarity_pass": score.similarity_pass,
        "nearest_selected_similarity": score.nearest_selected_similarity,
    }


def _restore_score(value: object) -> SelectionScore:
    score = _mapping(value, "score")
    return SelectionScore(
        base_utility=_number(score.get("base_utility"), "base_utility"),
        spoiler_penalty=_number(score.get("spoiler_penalty"), "spoiler_penalty"),
        coverage_bonus=_number(score.get("coverage_bonus"), "coverage_bonus"),
        temporal_diversity_penalty=_number(
            score.get("temporal_diversity_penalty"),
            "temporal_diversity_penalty",
        ),
        marginal_utility=_number(
            score.get("marginal_utility"),
            "marginal_utility",
        ),
        similarity_pass=_number(score.get("similarity_pass"), "similarity_pass"),
        nearest_selected_similarity=_optional_number(
            score.get("nearest_selected_similarity"),
            "nearest_selected_similarity",
        ),
    )


def _candidate(
    value: object,
    candidates: Mapping[str, BlogCandidate],
) -> BlogCandidate:
    if not isinstance(value, str) or value not in candidates:
        raise ValueError("Video Set Selection candidate IDが不正です")
    return candidates[value]


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"Video Set Selection {label}が不正です")
    return cast(dict[str, object], value)


def _integer_mapping(value: object, label: str) -> dict[str, int]:
    mapping = _mapping(value, label)
    if not mapping or any(
        not isinstance(item, int) or isinstance(item, bool) or item < 0
        for item in mapping.values()
    ):
        raise ValueError(f"Video Set Selection {label}が不正です")
    return cast(dict[str, int], mapping)


def _integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"Video Set Selection {label}が不正です")
    return value


def _number(value: object, label: str) -> float:
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise ValueError(f"Video Set Selection {label}が不正です")
    return float(value)


def _optional_number(value: object, label: str) -> float | None:
    return None if value is None else _number(value, label)


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("Video Set Selection optional IDが不正です")
    return value
