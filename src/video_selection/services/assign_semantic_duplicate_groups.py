"""画像の意味と動画内順序から重複候補の安定Groupを割り当てる。"""

import hashlib
import math
import unicodedata
from collections import defaultdict
from collections.abc import Iterable

from ..models.blog_candidate import BlogCandidate
from ..models.combat_subject_evidence import CombatSubjectEvidence
from ..models.semantic_duplicate_basis import SemanticDuplicateBasis

SemanticDuplicateGroupAssignments = tuple[
    dict[str, str],
    dict[str, SemanticDuplicateBasis],
    dict[str, tuple[str, ...]],
]

_COMBAT_ENCOUNTER_BASIS: SemanticDuplicateBasis = "combat_encounter_sequence"
_COMBAT_SUBJECT_BASIS: SemanticDuplicateBasis = "combat_subject_appearance"
_TITLE_SEMANTICS_BASIS: SemanticDuplicateBasis = "title_semantics"
_VISUAL_ROLE_SIMILARITY_BASIS: SemanticDuplicateBasis = "visual_role_similarity"
_MAX_ISOLATED_SCENE_BLIP_SECONDS = 15
_MAX_VISUAL_ROLE_DISTANCE_SECONDS = 30
_VISUAL_ROLE_SIMILARITY_THRESHOLD = 0.93
_COMBAT_SUBJECT_VISUAL_SIMILARITY_THRESHOLD = 0.80
_SEMANTIC_BASIS_PRIORITY: dict[SemanticDuplicateBasis, int] = {
    "combat_subject_appearance": 0,
    "combat_encounter_sequence": 1,
    "title_semantics": 2,
    "visual_role_similarity": 3,
}


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
    groups.extend(_combat_subject_groups(eligible_candidates))
    groups.extend(
        (_COMBAT_ENCOUNTER_BASIS, members)
        for members in assign_combat_encounter_groups(candidates)
    )
    groups.extend(_visual_role_similarity_groups(eligible_candidates))

    group_ids: dict[str, str] = {}
    bases: dict[str, SemanticDuplicateBasis] = {}
    evidence_by_member: dict[str, tuple[str, ...]] = {}
    for basis, members in _merge_overlapping_groups(groups):
        if len(members) < 2:
            continue
        member_ids = tuple(sorted(member.identifier for member in members))
        payload = "\0".join((basis, *member_ids))
        group_id = "semantic_" + hashlib.sha256(payload.encode()).hexdigest()
        group_evidence = _group_evidence(basis, members)
        for member_id in member_ids:
            if member_id in group_ids:
                raise ValueError("Semantic Duplicate Groupが重複しています")
            group_ids[member_id] = group_id
            bases[member_id] = basis
            if group_evidence:
                evidence_by_member[member_id] = group_evidence
    return group_ids, bases, evidence_by_member


def assign_combat_encounter_groups(
    candidates: tuple[BlogCandidate, ...],
) -> tuple[tuple[BlogCandidate, ...], ...]:
    """Semantic Group統合前のCombat Encounter Groupを返す。"""
    return tuple(
        members
        for _, members in _combat_encounter_groups(candidates)
        if len(members) >= 2
    )


def _group_evidence(
    basis: SemanticDuplicateBasis,
    members: tuple[BlogCandidate, ...],
) -> tuple[str, ...]:
    """Combat Subject Groupへ共通するprivacy-safeな有限enum根拠を返す。"""
    if basis != _COMBAT_SUBJECT_BASIS:
        return ()
    evidence = tuple(
        item.annotation.combat_subject_evidence
        for item in members
        if item.annotation.combat_subject_evidence is not None
        and item.annotation.combat_subject_evidence.can_identify_subject
    )
    if len(evidence) != len(members):
        return ()
    tokens: list[str] = []
    for field_name in ("body_plan", "scale", "surface"):
        values = {getattr(item, field_name) for item in evidence}
        if len(values) == 1:
            tokens.append(f"{field_name}:{next(iter(values))}")
    common_colors = set(evidence[0].colors).intersection(
        *(set(item.colors) for item in evidence[1:])
    )
    common_traits = set(evidence[0].traits).intersection(
        *(set(item.traits) for item in evidence[1:])
    )
    tokens.extend(f"color:{color}" for color in sorted(common_colors))
    tokens.extend(f"trait:{trait}" for trait in sorted(common_traits))
    return tuple(tokens)


def _merge_overlapping_groups(
    groups: list[tuple[SemanticDuplicateBasis, tuple[BlogCandidate, ...]]],
) -> tuple[tuple[SemanticDuplicateBasis, tuple[BlogCandidate, ...]], ...]:
    """公開根拠のmember境界を保って重なるSemantic Groupを整理する。"""
    parents: dict[str, str] = {}
    candidates_by_id: dict[str, BlogCandidate] = {}

    def find(identifier: str) -> str:
        parent = parents.setdefault(identifier, identifier)
        while parent != parents[parent]:
            parents[parent] = parents[parents[parent]]
            parent = parents[parent]
        parents[identifier] = parent
        return parent

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        lower, higher = sorted((left_root, right_root))
        parents[higher] = lower

    applicable_groups = tuple(
        (
            basis,
            tuple(sorted(members, key=lambda item: item.identifier)),
        )
        for basis, members in groups
        if len(members) >= 2
    )
    for _, members in applicable_groups:
        member_ids = tuple(member.identifier for member in members)
        for member in members:
            candidates_by_id[member.identifier] = member
            find(member.identifier)
        for member_id in member_ids[1:]:
            union(member_ids[0], member_id)

    members_by_root: dict[str, list[BlogCandidate]] = defaultdict(list)
    for identifier, candidate in candidates_by_id.items():
        members_by_root[find(identifier)].append(candidate)
    groups_by_root: dict[
        str,
        list[tuple[SemanticDuplicateBasis, tuple[BlogCandidate, ...]]],
    ] = defaultdict(list)
    for basis, members in applicable_groups:
        groups_by_root[find(members[0].identifier)].append((basis, members))

    merged: list[tuple[SemanticDuplicateBasis, tuple[BlogCandidate, ...]]] = []
    for root, component_members in members_by_root.items():
        ordered_members = tuple(
            sorted(component_members, key=lambda item: item.identifier)
        )
        merged.extend(
            _published_component_groups(groups_by_root[root], ordered_members)
        )
    return tuple(
        sorted(
            merged,
            key=lambda item: tuple(member.identifier for member in item[1]),
        )
    )


def _published_component_groups(
    originating_groups: list[tuple[SemanticDuplicateBasis, tuple[BlogCandidate, ...]]],
    component_members: tuple[BlogCandidate, ...],
) -> tuple[tuple[SemanticDuplicateBasis, tuple[BlogCandidate, ...]], ...]:
    """根拠が実際に説明するmemberだけを公開Groupとして返す。"""
    component_ids = tuple(member.identifier for member in component_members)
    spanning_groups = tuple(
        (basis, members)
        for basis, members in originating_groups
        if tuple(member.identifier for member in members) == component_ids
        and _basis_is_publishable(basis, members)
    )
    if spanning_groups:
        basis, _ = min(spanning_groups, key=_originating_group_sort_key)
        return ((basis, component_members),)

    published = []
    claimed_member_ids: set[str] = set()
    for basis, members in sorted(
        originating_groups,
        key=_originating_group_sort_key,
    ):
        member_ids = {member.identifier for member in members}
        if claimed_member_ids.isdisjoint(member_ids) and _basis_is_publishable(
            basis, members
        ):
            published.append((basis, members))
            claimed_member_ids.update(member_ids)
    return tuple(published)


def _originating_group_sort_key(
    group: tuple[SemanticDuplicateBasis, tuple[BlogCandidate, ...]],
) -> tuple[int, str, tuple[str, ...]]:
    """元Groupを公開優先度と安定member順へ正規化する。"""
    basis, members = group
    return (
        _SEMANTIC_BASIS_PRIORITY[basis],
        basis,
        tuple(member.identifier for member in members),
    )


def _basis_is_publishable(
    basis: SemanticDuplicateBasis,
    members: tuple[BlogCandidate, ...],
) -> bool:
    """元Groupのmember全体がbasisの公開contractを満たすかを返す。"""
    return basis != _COMBAT_SUBJECT_BASIS or bool(_group_evidence(basis, members))


def _combat_subject_groups(
    candidates: tuple[BlogCandidate, ...],
) -> Iterable[tuple[SemanticDuplicateBasis, tuple[BlogCandidate, ...]]]:
    """十分な外見根拠が一致する主要戦闘対象を動画横断でまとめる。"""
    unassigned = sorted(candidates, key=lambda item: item.identifier)
    while unassigned:
        root = unassigned.pop(0)
        component = [root]
        for other in tuple(unassigned):
            if all(_has_same_combat_subject(other, member) for member in component):
                component.append(other)
                unassigned.remove(other)
        if len(component) > 1:
            yield _COMBAT_SUBJECT_BASIS, tuple(component)


def _has_same_combat_subject(left: BlogCandidate, right: BlogCandidate) -> bool:
    """外見根拠とNeutral特徴が同じ主要戦闘対象を支持するかを返す。"""
    left_evidence = left.annotation.combat_subject_evidence
    right_evidence = right.annotation.combat_subject_evidence
    return (
        left.annotation.combat_encounter_kind == "major"
        and right.annotation.combat_encounter_kind == "major"
        and left_evidence is not None
        and right_evidence is not None
        and left_evidence.can_identify_subject
        and right_evidence.can_identify_subject
        and _combat_subject_evidence_matches(left_evidence, right_evidence)
        and _cosine_similarity(left, right)
        >= _COMBAT_SUBJECT_VISUAL_SIMILARITY_THRESHOLD
    )


def _combat_subject_evidence_matches(
    left: CombatSubjectEvidence,
    right: CombatSubjectEvidence,
) -> bool:
    """独立観測の一部列挙差を許容して外見の中核特徴を比較する。"""
    return (
        left.body_plan == right.body_plan
        and left.scale == right.scale
        and left.surface == right.surface
        and not set(left.colors).isdisjoint(right.colors)
        and not set(left.traits).isdisjoint(right.traits)
    )


def _has_title_semantics(candidate: BlogCandidate) -> bool:
    """分類名または画像内根拠がタイトル画面を示すかを返す。"""
    return candidate.annotation.has_title_semantics


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
            yield from _subject_aware_encounter_groups(encounter)
            encounter = []
        encounter.extend(run)
        current_slug = effective_slug
    if encounter:
        yield from _subject_aware_encounter_groups(encounter)


def _subject_aware_encounter_groups(
    candidates: list[BlogCandidate],
) -> Iterable[tuple[SemanticDuplicateBasis, tuple[BlogCandidate, ...]]]:
    """同じ遭遇内で明確に異なる戦闘対象を別Groupとして維持する。"""
    evidenced_candidates = [
        candidate
        for candidate in candidates
        if (evidence := candidate.annotation.combat_subject_evidence) is not None
        and evidence.distinctiveness != "unclear"
    ]
    identifiable_candidates = [
        candidate
        for candidate in evidenced_candidates
        if candidate.annotation.combat_subject_evidence is not None
        and candidate.annotation.combat_subject_evidence.can_identify_subject
    ]
    if not any(
        _have_incompatible_combat_subject_evidence(left, right)
        for index, left in enumerate(evidenced_candidates)
        for right in evidenced_candidates[index + 1 :]
    ):
        yield _COMBAT_ENCOUNTER_BASIS, tuple(candidates)
        return
    unassigned = sorted(identifiable_candidates, key=lambda item: item.identifier)
    while unassigned:
        root = unassigned.pop(0)
        component = [root]
        for other in tuple(unassigned):
            if all(
                _have_compatible_combat_subject_evidence(other, member)
                for member in component
            ):
                component.append(other)
                unassigned.remove(other)
        if len(component) > 1:
            yield _COMBAT_ENCOUNTER_BASIS, tuple(component)


def _have_incompatible_combat_subject_evidence(
    left: BlogCandidate,
    right: BlogCandidate,
) -> bool:
    """同一遭遇内で具体的な外見根拠が明確に競合するかを返す。"""
    left_evidence = left.annotation.combat_subject_evidence
    right_evidence = right.annotation.combat_subject_evidence
    if left_evidence is None or right_evidence is None:
        return False
    for field_name in ("body_plan", "scale", "surface"):
        left_value = getattr(left_evidence, field_name)
        right_value = getattr(right_evidence, field_name)
        if (
            left_value != "unknown"
            and right_value != "unknown"
            and left_value != right_value
        ):
            return True
    return (
        bool(left_evidence.colors)
        and bool(right_evidence.colors)
        and set(left_evidence.colors).isdisjoint(right_evidence.colors)
    ) or (
        bool(left_evidence.traits)
        and bool(right_evidence.traits)
        and set(left_evidence.traits).isdisjoint(right_evidence.traits)
    )


def _have_compatible_combat_subject_evidence(
    left: BlogCandidate,
    right: BlogCandidate,
) -> bool:
    """同一遭遇の時間的根拠を補強できる外見互換性を返す。"""
    left_evidence = left.annotation.combat_subject_evidence
    right_evidence = right.annotation.combat_subject_evidence
    return (
        left_evidence is not None
        and right_evidence is not None
        and left_evidence.can_identify_subject
        and right_evidence.can_identify_subject
        and _combat_subject_evidence_matches(left_evidence, right_evidence)
    )


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
