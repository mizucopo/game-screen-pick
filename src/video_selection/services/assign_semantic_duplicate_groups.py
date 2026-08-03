"""画像の意味と動画内順序から重複候補の安定Groupを割り当てる。"""

import hashlib
import math
import unicodedata
from collections import defaultdict
from collections.abc import Iterable

from ..models.blog_candidate import BlogCandidate
from ..models.semantic_duplicate_basis import SemanticDuplicateBasis

SemanticDuplicateGroupAssignments = tuple[
    dict[str, str],
    dict[str, SemanticDuplicateBasis],
]

_COMBAT_ENCOUNTER_BASIS: SemanticDuplicateBasis = "combat_encounter_sequence"
_TITLE_SEMANTICS_BASIS: SemanticDuplicateBasis = "title_semantics"
_VISUAL_ROLE_SIMILARITY_BASIS: SemanticDuplicateBasis = "visual_role_similarity"
_MAX_ISOLATED_SCENE_BLIP_SECONDS = 15
_MAX_VISUAL_ROLE_DISTANCE_SECONDS = 30
_VISUAL_ROLE_SIMILARITY_THRESHOLD = 0.93


def assign_semantic_duplicate_groups(
    candidates: tuple[BlogCandidate, ...],
) -> SemanticDuplicateGroupAssignments:
    """意味的に同時採用しない候補へ決定的なGroup IDと根拠を返す。"""
    eligible_candidates = tuple(
        candidate
        for candidate in candidates
        if candidate.annotation.explanation_value != "none"
    )
    groups: list[tuple[SemanticDuplicateBasis, tuple[BlogCandidate, ...]]] = []
    title_candidates = tuple(
        candidate
        for candidate in eligible_candidates
        if _has_title_semantics(candidate)
    )
    if title_candidates:
        groups.append((_TITLE_SEMANTICS_BASIS, title_candidates))
    groups.extend(_combat_encounter_groups(candidates))
    groups.extend(_visual_role_similarity_groups(eligible_candidates))

    group_ids: dict[str, str] = {}
    bases: dict[str, SemanticDuplicateBasis] = {}
    for basis, members in groups:
        if len(members) < 2:
            continue
        member_ids = tuple(sorted(member.identifier for member in members))
        payload = "\0".join((basis, *member_ids))
        group_id = "semantic_" + hashlib.sha256(payload.encode()).hexdigest()
        for member_id in member_ids:
            if member_id in group_ids:
                raise ValueError("Semantic Duplicate Groupが重複しています")
            group_ids[member_id] = group_id
            bases[member_id] = basis
    return group_ids, bases


def _has_title_semantics(candidate: BlogCandidate) -> bool:
    """分類名または画像内根拠がタイトル画面を示すかを返す。"""
    annotation = candidate.annotation
    evidence = annotation.representative_frame_evidence
    return (
        annotation.blog_image_type == "title"
        or annotation.screen_text_kind == "title"
        or evidence is not None
        and evidence.content_kind == "title"
    )


def _combat_encounter_groups(
    candidates: tuple[BlogCandidate, ...],
) -> Iterable[tuple[SemanticDuplicateBasis, tuple[BlogCandidate, ...]]]:
    """同一動画内の連続した主要戦闘を一つの遭遇として返す。"""
    by_source: dict[tuple[int, str], list[BlogCandidate]] = defaultdict(list)
    for candidate in candidates:
        fingerprint = candidate.annotation.candidate.video_fingerprint
        if fingerprint is not None:
            by_source[(candidate.video_order, fingerprint)].append(candidate)

    for source_key in sorted(by_source):
        ordered = sorted(
            by_source[source_key],
            key=lambda item: (
                item.annotation.candidate.video_time,
                item.identifier,
            ),
        )
        major_block: list[BlogCandidate] = []
        for candidate in ordered:
            annotation = candidate.annotation
            if annotation.combat_encounter_kind == "major" and not _has_title_semantics(
                candidate
            ):
                if annotation.explanation_value != "none":
                    major_block.append(candidate)
                continue
            yield from _groups_within_major_block(major_block)
            major_block = []
        yield from _groups_within_major_block(major_block)


def _groups_within_major_block(
    candidates: list[BlogCandidate],
) -> Iterable[tuple[SemanticDuplicateBasis, tuple[BlogCandidate, ...]]]:
    """非主要場面で区切られたblockをScene Slugごとの遭遇へ分ける。"""
    runs = _scene_runs(candidates)
    effective_slugs = [run[0].annotation.scene_slug for run in runs]
    for index in range(1, len(runs) - 1):
        if (
            len(runs[index]) == 1
            and effective_slugs[index - 1] == effective_slugs[index + 1]
            and _is_nearby_scene_blip(
                runs[index - 1][-1],
                runs[index][0],
                runs[index + 1][0],
            )
        ):
            effective_slugs[index] = effective_slugs[index - 1]

    encounter: list[BlogCandidate] = []
    current_slug: str | None = None
    for run, effective_slug in zip(runs, effective_slugs, strict=True):
        if encounter and effective_slug != current_slug:
            yield _COMBAT_ENCOUNTER_BASIS, tuple(encounter)
            encounter = []
        encounter.extend(run)
        current_slug = effective_slug
    if encounter:
        yield _COMBAT_ENCOUNTER_BASIS, tuple(encounter)


def _visual_role_similarity_groups(
    candidates: tuple[BlogCandidate, ...],
) -> Iterable[tuple[SemanticDuplicateBasis, tuple[BlogCandidate, ...]]]:
    """近接し同じ画像内役割を持つ高類似候補をGroupとして返す。"""
    by_source: dict[tuple[int, str], list[BlogCandidate]] = defaultdict(list)
    for candidate in candidates:
        annotation = candidate.annotation
        fingerprint = annotation.candidate.video_fingerprint
        if (
            fingerprint is not None
            and annotation.combat_encounter_kind != "major"
            and not _has_title_semantics(candidate)
            and annotation.representative_frame_evidence is not None
        ):
            by_source[(candidate.video_order, fingerprint)].append(candidate)

    for source_key in sorted(by_source):
        unassigned = sorted(
            by_source[source_key],
            key=lambda item: (
                item.annotation.candidate.video_time,
                item.identifier,
            ),
        )
        while unassigned:
            root = unassigned.pop(0)
            component = [root]
            for other in tuple(unassigned):
                if all(
                    _has_same_nearby_visual_role(other, member) for member in component
                ):
                    component.append(other)
                    unassigned.remove(other)
            if len(component) > 1:
                yield _VISUAL_ROLE_SIMILARITY_BASIS, tuple(component)


def _has_same_nearby_visual_role(
    left: BlogCandidate,
    right: BlogCandidate,
) -> bool:
    """画像内役割、時刻、Neutral特徴が意味的重複を支持するかを返す。"""
    left_annotation = left.annotation
    right_annotation = right.annotation
    left_evidence = left_annotation.representative_frame_evidence
    right_evidence = right_annotation.representative_frame_evidence
    left_time = left_annotation.candidate.video_time
    right_time = right_annotation.candidate.video_time
    return (
        left_evidence is not None
        and right_evidence is not None
        and left_evidence.content_kind == right_evidence.content_kind
        and left_annotation.combat_encounter_kind
        == right_annotation.combat_encounter_kind
        and (
            left.scene_selection_role != "recurring_gameplay"
            and right.scene_selection_role != "recurring_gameplay"
            or _semantic_descriptions_match(left, right)
        )
        and left_time is not None
        and right_time is not None
        and abs(left_time - right_time) <= _MAX_VISUAL_ROLE_DISTANCE_SECONDS
        and _cosine_similarity(left, right) >= _VISUAL_ROLE_SIMILARITY_THRESHOLD
    )


def _semantic_descriptions_match(
    left: BlogCandidate,
    right: BlogCandidate,
) -> bool:
    """独立評価された画像説明が同じ追加説明価値を示すかを返す。"""
    left_summary = _normalized_semantic_text(left.annotation.summary)
    right_summary = _normalized_semantic_text(right.annotation.summary)
    return bool(left_summary) and left_summary == right_summary


def _normalized_semantic_text(value: str) -> str:
    """表記差を除いた画像説明の比較値を返す。"""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(
        character
        for character in normalized
        if unicodedata.category(character)[0] in {"L", "N"}
    )


def _cosine_similarity(left: BlogCandidate, right: BlogCandidate) -> float:
    """Neutral Image Analysisのvisual feature間cosine similarityを返す。"""
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


def _scene_runs(
    candidates: list[BlogCandidate],
) -> list[list[BlogCandidate]]:
    """時系列候補を連続するScene Slugごとのrunへ分割する。"""
    runs: list[list[BlogCandidate]] = []
    for candidate in candidates:
        if (
            not runs
            or runs[-1][-1].annotation.scene_slug != candidate.annotation.scene_slug
        ):
            runs.append([])
        runs[-1].append(candidate)
    return runs


def _is_nearby_scene_blip(
    previous: BlogCandidate,
    candidate: BlogCandidate,
    following: BlogCandidate,
) -> bool:
    """前後15秒以内に限り単発のScene Slug揺れと判断する。"""
    previous_time = previous.annotation.candidate.video_time
    candidate_time = candidate.annotation.candidate.video_time
    following_time = following.annotation.candidate.video_time
    if previous_time is None or candidate_time is None or following_time is None:
        return False
    return (
        candidate_time - previous_time <= _MAX_ISOLATED_SCENE_BLIP_SECONDS
        and following_time - candidate_time <= _MAX_ISOLATED_SCENE_BLIP_SECONDS
    )
