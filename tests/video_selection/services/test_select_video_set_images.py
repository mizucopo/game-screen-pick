"""決定的なVideo Set selectorのtest。"""

import hashlib
import math
from fractions import Fraction
from typing import Literal

import cv2
import numpy as np
import pytest

from src.video_selection.models.blog_candidate import BlogCandidate
from src.video_selection.models.candidate_annotation import (
    BlogImageType,
    CandidateAnnotation,
    ContextCueRelevance,
    ExplanationValue,
    ScreenTextKind,
    SelectionCoverageFacet,
    SpoilerRisk,
)
from src.video_selection.models.combat_encounter_basis import CombatEncounterBasis
from src.video_selection.models.combat_encounter_kind import CombatEncounterKind
from src.video_selection.models.combat_subject_evidence import CombatSubjectEvidence
from src.video_selection.models.decoded_video_frame import DecodedVideoFrame
from src.video_selection.models.frame_candidate import FrameCandidate
from src.video_selection.models.neutral_image_analysis import NeutralImageAnalysis
from src.video_selection.models.neutral_image_metrics import NeutralImageMetrics
from src.video_selection.models.representative_frame_evidence import (
    CandidateFrameContentKind,
    RepresentativeFrameEvidence,
)
from src.video_selection.models.scene_catalog_entry import SceneSelectionRole
from src.video_selection.models.video_set_selection_result import (
    VideoSetSelectionResult,
)
from src.video_selection.services.analyze_neutral_images import (
    analyze_neutral_images,
)
from src.video_selection.services.select_video_set_images import (
    SpoilerSensitivity,
    select_completable_coverage_prerequisites,
    select_from_shortlist_batches,
    select_video_set_images,
)

type CandidateSpec = tuple[
    str,
    float,
    tuple[float, ...],
    Fraction,
    BlogImageType,
    ExplanationValue,
    SpoilerRisk,
]

_COMBAT_ENCOUNTER_BASIS_BY_KIND: dict[
    CombatEncounterKind,
    CombatEncounterBasis,
] = {
    "not_combat": "none",
    "ordinary": "ordinary_opponent_presentation",
    "major": "major_opponent_presentation",
    "uncertain": "ambiguous",
}


def _metrics() -> NeutralImageMetrics:
    return NeutralImageMetrics(
        blur_score=100.0,
        brightness=100.0,
        contrast=50.0,
        edge_density=0.2,
        color_richness=0.5,
        ui_density=0.2,
        action_intensity=0.4,
        visual_balance=0.8,
        dramatic_score=0.3,
        luminance_entropy=1.0,
        luminance_range=100.0,
        near_black_ratio=0.0,
        near_white_ratio=0.0,
        dominant_tone_ratio=0.2,
        information_score=0.8,
        visibility_score=0.9,
    )


def _decoded_frame(source_pts: int, rgb: np.ndarray) -> DecodedVideoFrame:
    height, width = rgb.shape[:2]
    return DecodedVideoFrame(
        stream_index=0,
        pts=source_pts,
        duration_ts=1,
        time_base=Fraction(1, 10),
        width=width,
        height=height,
        pixel_format="rgb24",
        pixels=rgb.astype(np.uint8).tobytes(),
    )


def _candidate(
    digest_character: str,
    *,
    quality: float,
    feature: tuple[float, ...],
    progress: Fraction,
    blog_image_type: BlogImageType,
    explanation_value: ExplanationValue,
    context_relevance: ContextCueRelevance,
    spoiler_risk: SpoilerRisk = "none",
    scene_selection_role: SceneSelectionRole = "ordinary",
    scene_slug: str | None = None,
    video_order: int = 0,
    combat_encounter_kind: CombatEncounterKind = "not_combat",
    video_fingerprint: str | None = None,
    screen_text_kind: ScreenTextKind = "none",
    content_kind: CandidateFrameContentKind | None = None,
    summary: str | None = None,
    combat_subject_evidence: CombatSubjectEvidence | None = None,
) -> BlogCandidate:
    digest = (
        digest_character * 64
        if len(digest_character) == 1 and digest_character in "0123456789abcdef"
        else hashlib.sha256(digest_character.encode()).hexdigest()
    )
    frame = FrameCandidate(
        identifier="frm_" + digest,
        image_bytes=digest_character.encode(),
        video_fingerprint=video_fingerprint or digest,
        stream_index=0,
        source_pts=int(progress * 1000),
        origin_pts=0,
        time_base=Fraction(1, 1000),
        video_time=progress * 100,
        analysis=NeutralImageAnalysis(
            source_pts=int(progress * 1000),
            metrics=_metrics(),
            quality_score=quality,
            visual_feature=feature,
            grayscale_signature=b"signature",
            reject_reason=None,
        ),
    )
    annotation = CandidateAnnotation(
        candidate=frame,
        candidate_moment_id="mom_" + digest,
        summary=summary or f"candidate {digest_character}",
        scene_slug=scene_slug or "scene-" + digest_character,
        blog_image_type=blog_image_type,
        explanation_value=explanation_value,
        context_relevance=context_relevance,
        supporting_context_cue_ids=(
            ("cue_" + digest,) if context_relevance in {"weak", "strong"} else ()
        ),
        spoiler_risk=spoiler_risk,
        spoiler_evidence=(
            "重大な物語情報が画像に示される" if spoiler_risk != "none" else ""
        ),
        combat_encounter_kind=combat_encounter_kind,
        combat_encounter_basis=_COMBAT_ENCOUNTER_BASIS_BY_KIND[combat_encounter_kind],
        combat_subject_evidence=combat_subject_evidence,
        screen_text_kind=screen_text_kind,
        representative_frame_evidence=(
            RepresentativeFrameEvidence(
                content_kind=content_kind,
                primary_subject_visibility="clear",
                opponent_body_visibility="absent",
                transient_obstruction="none",
            )
            if content_kind is not None
            else None
        ),
    )
    return BlogCandidate(
        annotation=annotation,
        scene_selection_role=scene_selection_role,
        video_order=video_order,
        video_set_progress=progress,
        shortlist_rank=(
            ord(digest_character) if len(digest_character) == 1 else int(digest[:8], 16)
        ),
    )


def _candidate_moment_timelines(
    candidates: tuple[BlogCandidate, ...],
) -> dict[tuple[int, str], tuple[str, ...]]:
    """候補集合からsource別の完全なCandidate Moment時系列を返す。"""
    grouped: dict[tuple[int, str], list[BlogCandidate]] = {}
    for candidate in candidates:
        fingerprint = candidate.annotation.candidate.video_fingerprint
        assert fingerprint is not None
        grouped.setdefault((candidate.video_order, fingerprint), []).append(candidate)
    timelines: dict[tuple[int, str], tuple[str, ...]] = {}
    for key, values in grouped.items():
        moment_ids: list[str] = []
        for item in sorted(
            values,
            key=lambda item: (
                item.annotation.candidate.video_time,
                item.annotation.candidate_moment_id,
            ),
        ):
            moment_id = item.annotation.candidate_moment_id
            assert moment_id is not None
            moment_ids.append(moment_id)
        timelines[key] = tuple(moment_ids)
    return timelines


def _hard_limit_variant_candidates(
    hard_limit: Literal["title", "spoiler"],
) -> tuple[BlogCandidate, ...]:
    """hard limitで相互排他になるVariant Group候補集合を返す。"""
    feature_count = 19

    def unit_feature(index: int) -> tuple[float, ...]:
        return tuple(float(position == index) for position in range(feature_count))

    repeated_combat = _candidate(
        f"repeated-combat-{hard_limit}",
        quality=0.9,
        feature=unit_feature(0),
        progress=Fraction(1, 100),
        blog_image_type="normal_gameplay",
        explanation_value="high",
        context_relevance="none",
        scene_selection_role="recurring_gameplay",
        scene_slug="battle",
        combat_encounter_kind="ordinary",
    )
    event = _candidate(
        f"event-{hard_limit}",
        quality=0.8,
        feature=(0.96, math.sqrt(1 - 0.96**2), *(0.0 for _ in range(17))),
        progress=Fraction(2, 100),
        blog_image_type="event",
        explanation_value="high",
        context_relevance="none",
        scene_selection_role="recurring_gameplay",
        scene_slug="battle",
    )
    alternative_combat = _candidate(
        f"alternative-combat-{hard_limit}",
        quality=0.1,
        feature=unit_feature(2),
        progress=Fraction(3, 100),
        blog_image_type="normal_gameplay",
        explanation_value="high",
        context_relevance="none",
        combat_encounter_kind="ordinary",
    )
    prerequisite_groups = tuple(
        _candidate(
            f"{hard_limit}-group-{index}",
            quality=0.1,
            feature=unit_feature(index),
            progress=Fraction(index + 1, 100),
            blog_image_type=("title" if hard_limit == "title" else "normal_gameplay"),
            explanation_value="high",
            context_relevance="none",
            spoiler_risk="high" if hard_limit == "spoiler" else "none",
            scene_selection_role="recurring_gameplay",
            scene_slug="battle",
        )
        for index in range(3, 12)
    )
    fillers = tuple(
        _candidate(
            f"{hard_limit}-filler-{index}",
            quality=1.0,
            feature=unit_feature(index),
            progress=Fraction(index + 1, 100),
            blog_image_type="normal_gameplay",
            explanation_value="high",
            context_relevance="none",
        )
        for index in range(12, 19)
    )
    return (
        repeated_combat,
        event,
        alternative_combat,
        *prerequisite_groups,
        *fillers,
    )


def _normalized(result: VideoSetSelectionResult) -> list[dict[str, object]]:
    selected = result.selected
    return [
        {
            "id": item.candidate.annotation.candidate.identifier,
            "reason_codes": list(item.reason_codes),
            "base": round(item.score.base_utility, 6),
            "coverage": round(item.score.coverage_bonus, 6),
            "spoiler": round(item.score.spoiler_penalty, 6),
            "temporal": round(item.score.temporal_diversity_penalty, 6),
            "marginal": round(item.score.marginal_utility, 6),
            "pass": item.score.similarity_pass,
        }
        for item in selected
    ]


def test_normalized_selection_is_exact_and_independent_of_input_order() -> None:
    """同じ候補集合のselected ID・順序・理由・数値が一定であること。

    Arrange:
        - 品質、説明価値、coverage、spoiler、進行位置が異なる4候補が用意される
        - 同じ候補集合が異なる入力順で用意される
    Act:
        - Video Set selectorが両方の候補集合へ実行される
    Assert:
        - 正規化したselected ID、順序、reason code、数値内訳がgoldenと一致すること
    """
    # Arrange
    candidates = (
        _candidate(
            "b",
            quality=0.9,
            feature=(1.0, 0.0, 0.0, 0.0),
            progress=Fraction(82, 100),
            blog_image_type="normal_gameplay",
            explanation_value="high",
            context_relevance="strong",
        ),
        _candidate(
            "d",
            quality=0.8,
            feature=(0.0, 1.0, 0.0, 0.0),
            progress=Fraction(40, 100),
            blog_image_type="event",
            explanation_value="high",
            context_relevance="strong",
        ),
        _candidate(
            "c",
            quality=0.95,
            feature=(0.0, 0.0, 1.0, 0.0),
            progress=Fraction(90, 100),
            blog_image_type="event",
            explanation_value="high",
            context_relevance="strong",
            spoiler_risk="high",
        ),
        _candidate(
            "a",
            quality=0.7,
            feature=(0.0, 0.0, 0.0, 1.0),
            progress=Fraction(10, 100),
            blog_image_type="normal_gameplay",
            explanation_value="low",
            context_relevance="none",
        ),
    )
    expected = [
        {
            "id": "frm_" + "b" * 64,
            "reason_codes": [
                "high_quality",
                "high_explanation_value",
                "strong_context_relevance",
                "normal_gameplay_coverage",
            ],
            "base": 0.93,
            "coverage": 0.1,
            "spoiler": 0.0,
            "temporal": 0.0,
            "marginal": 1.03,
            "pass": 0.72,
        },
        {
            "id": "frm_" + "d" * 64,
            "reason_codes": [
                "high_quality",
                "high_explanation_value",
                "strong_context_relevance",
                "event_coverage",
            ],
            "base": 0.86,
            "coverage": 0.1,
            "spoiler": 0.0,
            "temporal": 0.0,
            "marginal": 0.96,
            "pass": 0.72,
        },
        {
            "id": "frm_" + "c" * 64,
            "reason_codes": [
                "high_quality",
                "high_explanation_value",
                "strong_context_relevance",
                "high_spoiler_penalty_applied",
            ],
            "base": 0.965,
            "coverage": 0.0,
            "spoiler": 0.1,
            "temporal": 0.0544,
            "marginal": 0.8106,
            "pass": 0.72,
        },
        {
            "id": "frm_" + "a" * 64,
            "reason_codes": ["normal_gameplay_coverage"],
            "base": 0.573333,
            "coverage": 0.1,
            "spoiler": 0.0,
            "temporal": 0.0,
            "marginal": 0.673333,
            "pass": 0.72,
        },
    ]

    # Act
    forward = select_video_set_images(
        candidates,
        requested_count=4,
        spoiler_sensitivity="medium",
        similarity_threshold=0.72,
    )
    reordered = select_video_set_images(
        tuple(reversed(candidates)),
        requested_count=4,
        spoiler_sensitivity="medium",
        similarity_threshold=0.72,
    )

    # Assert
    assert _normalized(forward) == expected
    assert _normalized(reordered) == expected


def test_visual_near_duplicate_is_rejected_even_when_selection_is_short() -> None:
    """Visual Near-Duplicateで要求不足が穴埋めされないこと。

    Arrange:
        - cosine similarityが0.995を超えるrecurring gameplay候補が2件用意される
        - 2枚の選定が要求される
    Act:
        - Video Set selectorが終端similarity passまで実行される
    Assert:
        - 近似重複の2件目が選択されずSelection Shortfallになること
        - stable rejection codeとblocking candidateが返されること
    """
    # Arrange
    first = _candidate(
        "e",
        quality=0.9,
        feature=(1.0, 0.0),
        progress=Fraction(1, 10),
        blog_image_type="normal_gameplay",
        explanation_value="high",
        context_relevance="none",
        scene_selection_role="recurring_gameplay",
    )
    second = _candidate(
        "f",
        quality=0.8,
        feature=(0.996, math.sqrt(1 - 0.996**2)),
        progress=Fraction(8, 10),
        blog_image_type="normal_gameplay",
        explanation_value="high",
        context_relevance="none",
        scene_selection_role="recurring_gameplay",
    )

    # Act
    result = select_video_set_images(
        (second, first),
        requested_count=2,
        spoiler_sensitivity="medium",
        similarity_threshold=0.72,
    )

    # Assert
    assert [item.candidate.identifier for item in result.selected] == [first.identifier]
    assert result.shortfall is True
    assert result.final_similarity_ceiling == 0.97
    assert len(result.rejected) == 1
    rejection = result.rejected[0]
    assert rejection.candidate.identifier == second.identifier
    assert rejection.reason_code == "visual_near_duplicate"
    assert rejection.nearest_selected_image_id == first.identifier
    assert rejection.similarity == pytest.approx(0.996)


def test_analyzed_distinct_types_coexist_without_duplicate_filling_shortfall() -> None:
    """異なるgameplayとmenuが共存し重複画像で穴埋めされないこと。

    Arrange:
        - edge方向分布は近いが色・輝度・配置が異なる2種類のframeが用意される
        - 各frameが有用なgameplay候補とmenu候補に注釈される
        - gameplayと同じ画素を持つ重複候補が用意される
    Act:
        - Neutral Image Analysisの視覚特徴でVideo Set selectorが実行される
    Assert:
        - 2候補がVisual Near-Duplicateにされず同時に選択されること
        - 重複候補が要求枚数の不足を埋めないこと
    """
    # Arrange
    rows, columns = np.indices((256, 256))
    gameplay_mask = ((rows // 32 + columns // 32) % 2).astype(bool)
    gameplay_rgb = np.empty((256, 256, 3), dtype=np.uint8)
    gameplay_rgb[gameplay_mask] = (210, 45, 35)
    gameplay_rgb[~gameplay_mask] = (25, 70, 180)
    menu_rgb = np.full((256, 256, 3), (25, 90, 45), dtype=np.uint8)
    for center_y in (24, 72, 120, 168, 216):
        for center_x in (24, 72, 120, 168, 216):
            cv2.rectangle(
                menu_rgb,
                (center_x - 12, center_y - 12),
                (center_x + 12, center_y + 12),
                (235, 210, 80),
                -1,
            )
    gameplay_analysis, menu_analysis = analyze_neutral_images(
        (
            _decoded_frame(0, gameplay_rgb),
            _decoded_frame(1, menu_rgb),
        )
    )
    duplicate_analysis = analyze_neutral_images(
        (_decoded_frame(2, gameplay_rgb.copy()),)
    )[0]
    gameplay = _candidate(
        "1",
        quality=0.9,
        feature=gameplay_analysis.visual_feature,
        progress=Fraction(1, 10),
        blog_image_type="normal_gameplay",
        explanation_value="high",
        context_relevance="none",
    )
    menu = _candidate(
        "2",
        quality=0.8,
        feature=menu_analysis.visual_feature,
        progress=Fraction(8, 10),
        blog_image_type="menu",
        explanation_value="high",
        context_relevance="none",
    )
    duplicate = _candidate(
        "3",
        quality=0.7,
        feature=duplicate_analysis.visual_feature,
        progress=Fraction(9, 10),
        blog_image_type="normal_gameplay",
        explanation_value="high",
        context_relevance="none",
    )

    # Act
    result = select_video_set_images(
        (menu, duplicate, gameplay),
        requested_count=3,
        spoiler_sensitivity="medium",
        similarity_threshold=0.72,
    )

    # Assert
    assert [item.candidate.identifier for item in result.selected] == [
        gameplay.identifier,
        menu.identifier,
    ]
    assert result.shortfall is True
    assert result.blog_image_type_actuals["normal_gameplay"] == 1
    assert result.blog_image_type_actuals["menu"] == 1
    assert len(result.rejected) == 1
    assert result.rejected[0].candidate.identifier == duplicate.identifier
    assert result.rejected[0].reason_code == "visual_near_duplicate"


def test_automatic_relaxation_does_not_select_redundant_gameplay_frame() -> None:
    """自動緩和で視覚的に冗長なgameplay画像が穴埋めされないこと。

    Arrange:
        - 同じ戦闘をほぼ同じ構図で示すsimilarity 0.973の2候補が用意される
        - 組み込み既定値から2枚の選定が要求される
    Act:
        - selectorが自動similarity passの終端まで実行される
    Assert:
        - 2枚目が選択されず、0.97の終端でSelection Shortfallになること
    """
    # Arrange
    first = _candidate(
        "7",
        quality=0.9,
        feature=(1.0, 0.0),
        progress=Fraction(1, 10),
        blog_image_type="normal_gameplay",
        explanation_value="high",
        context_relevance="none",
        scene_selection_role="recurring_gameplay",
        scene_slug="battle",
    )
    redundant = _candidate(
        "8",
        quality=0.8,
        feature=(0.973, math.sqrt(1 - 0.973**2)),
        progress=Fraction(8, 10),
        blog_image_type="normal_gameplay",
        explanation_value="high",
        context_relevance="none",
        scene_selection_role="recurring_gameplay",
        scene_slug="battle",
    )

    # Act
    result = select_video_set_images(
        (redundant, first),
        requested_count=2,
        spoiler_sensitivity="medium",
        similarity_threshold=0.72,
    )

    # Assert
    assert [item.candidate.identifier for item in result.selected] == [first.identifier]
    assert result.shortfall is True
    assert result.final_similarity_ceiling == 0.97
    assert result.rejected[0].reason_code == "similarity_ceiling"
    assert result.rejected[0].similarity == pytest.approx(0.973)


def test_candidate_without_explanation_value_does_not_fill_shortfall() -> None:
    """説明価値のない候補で要求枚数が穴埋めされないこと。

    Arrange:
        - 説明価値がある候補と、高品質でも説明価値がない候補が用意される
        - 2枚の選定が要求される
    Act:
        - Video Set selectorが終端similarity passまで実行される
    Assert:
        - 説明価値がある候補だけが選択されSelection Shortfallになること
        - 説明価値がない候補がstable reason付きで未採用になること
    """
    # Arrange
    meaningful = _candidate(
        "9",
        quality=0.7,
        feature=(1.0, 0.0),
        progress=Fraction(1, 10),
        blog_image_type="normal_gameplay",
        explanation_value="medium",
        context_relevance="none",
    )
    meaningless = _candidate(
        "a",
        quality=0.99,
        feature=(0.0, 1.0),
        progress=Fraction(9, 10),
        blog_image_type="other",
        explanation_value="none",
        context_relevance="strong",
    )

    # Act
    result = select_video_set_images(
        (meaningless, meaningful),
        requested_count=2,
        spoiler_sensitivity="medium",
        similarity_threshold=0.72,
    )

    # Assert
    assert [item.candidate.identifier for item in result.selected] == [
        meaningful.identifier
    ]
    assert result.shortfall is True
    assert len(result.rejected) == 1
    assert result.rejected[0].candidate.identifier == meaningless.identifier
    assert result.rejected[0].reason_code == "lower_marginal_utility"


def test_recurring_gameplay_expands_only_after_each_variant_group() -> None:
    """recurring gameplayで各Variant Groupの代表後に状態差が選ばれること。

    Arrange:
        - 同じrecurring sceneに同一groupの高品質2候補と別groupの低品質候補がある
        - base ceilingでは最初の候補以外が視覚条件を満たさない
    Act:
        - 3枚の選定が要求され終端passまでselectorが実行される
    Assert:
        - 別groupの代表が同一groupの2枚目より先に選択されること
        - 同一groupの2枚目にvariant expansionのstable reasonが付くこと
    """
    # Arrange
    first = _candidate(
        "1",
        quality=0.9,
        feature=(1.0, 0.0, 0.0),
        progress=Fraction(1, 10),
        blog_image_type="normal_gameplay",
        explanation_value="high",
        context_relevance="none",
        scene_selection_role="recurring_gameplay",
        scene_slug="battle",
    )
    same_group = _candidate(
        "2",
        quality=0.89,
        feature=(0.96, math.sqrt(1 - 0.96**2), 0.0),
        progress=Fraction(8, 10),
        blog_image_type="normal_gameplay",
        explanation_value="high",
        context_relevance="none",
        scene_selection_role="recurring_gameplay",
        scene_slug="battle",
    )
    other_group = _candidate(
        "3",
        quality=0.5,
        feature=(0.9, 0.0, math.sqrt(1 - 0.9**2)),
        progress=Fraction(45, 100),
        blog_image_type="normal_gameplay",
        explanation_value="low",
        context_relevance="none",
        scene_selection_role="recurring_gameplay",
        scene_slug="battle",
    )

    # Act
    result = select_video_set_images(
        (same_group, other_group, first),
        requested_count=3,
        spoiler_sensitivity="medium",
        similarity_threshold=0.72,
    )

    # Assert
    assert [item.candidate.identifier for item in result.selected] == [
        first.identifier,
        other_group.identifier,
        same_group.identifier,
    ]
    first_selected, other_selected, expanded = result.selected
    assert first_selected.variant_group_id == expanded.variant_group_id
    assert other_selected.variant_group_id != expanded.variant_group_id
    assert expanded.score.similarity_pass == 0.97
    assert "recurring_gameplay_variant" in expanded.reason_codes


def test_spoiler_guarded_group_does_not_block_recurring_variant_expansion() -> None:
    """Spoiler上限で採用不能なGroupにより採用可能なvariantが阻害されないこと。

    Arrange:
        - 同じrecurring sceneに選択済みGroupのvariantと未代表のMajor Spoiler Groupがある
        - 別sceneのMajor Spoilerでlow感度由来の件数上限が満たされる候補順がある
    Act:
        - medium感度で3枚の選定が要求される
    Assert:
        - guard対象Groupを待たず採用可能なvariantで要求数が満たされること
    """
    # Arrange
    first = _candidate(
        "a",
        quality=0.8,
        feature=(1.0, 0.0, 0.0),
        progress=Fraction(1, 10),
        blog_image_type="normal_gameplay",
        explanation_value="low",
        context_relevance="none",
        scene_selection_role="recurring_gameplay",
        scene_slug="battle",
    )
    same_group = _candidate(
        "b",
        quality=0.8,
        feature=(0.96, math.sqrt(1 - 0.96**2), 0.0),
        progress=Fraction(9, 10),
        blog_image_type="normal_gameplay",
        explanation_value="low",
        context_relevance="none",
        scene_selection_role="recurring_gameplay",
        scene_slug="battle",
    )
    guarded_group = _candidate(
        "c",
        quality=127 / 140,
        feature=(0.0, 1.0, 0.0),
        progress=Fraction(1, 10),
        blog_image_type="event",
        explanation_value="low",
        context_relevance="none",
        spoiler_risk="high",
        scene_selection_role="recurring_gameplay",
        scene_slug="battle",
    )
    selected_major = _candidate(
        "d",
        quality=123 / 140,
        feature=(0.0, 0.97, math.sqrt(1 - 0.97**2)),
        progress=Fraction(9, 10),
        blog_image_type="event",
        explanation_value="low",
        context_relevance="none",
        spoiler_risk="high",
    )

    # Act
    result = select_video_set_images(
        (same_group, guarded_group, selected_major, first),
        requested_count=3,
        spoiler_sensitivity="medium",
        similarity_threshold=0.72,
    )

    # Assert
    assert [item.candidate.identifier for item in result.selected] == [
        first.identifier,
        selected_major.identifier,
        same_group.identifier,
    ]
    assert result.shortfall is False
    assert result.major_spoiler_limit == 1
    assert result.major_spoiler_selected_count == 1


def test_second_title_is_rejected_with_counterfactual_score() -> None:
    """2枚目のtitleがnear-miss数値を保ったままhard limitで拒否されること。

    Arrange:
        - 視覚的に異なるtitle候補が2件用意される
        - 2枚の選定が要求される
    Act:
        - Video Set selectorが終端passまで実行される
    Assert:
        - titleは1件だけ選択されSelection Shortfallになること
        - 2件目へsemantic duplicate、blocking ID、制約前の数値内訳が返されること
    """
    # Arrange
    best = _candidate(
        "4",
        quality=0.9,
        feature=(1.0, 0.0),
        progress=Fraction(1, 10),
        blog_image_type="title",
        explanation_value="high",
        context_relevance="none",
    )
    second = _candidate(
        "5",
        quality=0.8,
        feature=(0.0, 1.0),
        progress=Fraction(9, 10),
        blog_image_type="title",
        explanation_value="high",
        context_relevance="none",
    )

    # Act
    result = select_video_set_images(
        (second, best),
        requested_count=2,
        spoiler_sensitivity="medium",
        similarity_threshold=0.72,
    )

    # Assert
    assert [item.candidate.identifier for item in result.selected] == [best.identifier]
    assert result.shortfall is True
    assert len(result.rejected) == 1
    rejection = result.rejected[0]
    assert rejection.reason_code == "semantic_duplicate"
    assert rejection.blocked_by_image_id == best.identifier
    assert rejection.semantic_group_basis == "title_semantics"
    assert rejection.nearest_selected_image_id is None
    assert rejection.counterfactual_score.base_utility == pytest.approx(0.81)
    assert rejection.counterfactual_score.coverage_bonus == 0.05
    assert rejection.counterfactual_score.temporal_diversity_penalty == 0.0
    assert rejection.counterfactual_score.marginal_utility == pytest.approx(0.86)


def test_soft_coverage_allows_event_overflow_when_other_types_are_absent() -> None:
    """候補不足のtypeをhard quotaにせず有用なeventで超過できること。

    Arrange:
        - 視覚的に異なるevent候補だけが3件用意される
        - 3枚の選定が要求される
    Act:
        - Video Set selectorが実行される
    Assert:
        - event目標1件を超えて3件すべて選択されること
        - 5種のtargetとactualからcoverage超過が説明できること
    """
    # Arrange
    candidates = tuple(
        _candidate(
            digest,
            quality=quality,
            feature=feature,
            progress=progress,
            blog_image_type="event",
            explanation_value="high",
            context_relevance="none",
        )
        for digest, quality, feature, progress in (
            ("6", 0.9, (1.0, 0.0, 0.0), Fraction(1, 10)),
            ("7", 0.8, (0.0, 1.0, 0.0), Fraction(5, 10)),
            ("8", 0.7, (0.0, 0.0, 1.0), Fraction(9, 10)),
        )
    )

    # Act
    result = select_video_set_images(
        candidates,
        requested_count=3,
        spoiler_sensitivity="medium",
        similarity_threshold=0.72,
    )

    # Assert
    assert len(result.selected) == 3
    assert result.shortfall is False
    assert result.blog_image_type_targets == {
        "normal_gameplay": 2,
        "event": 1,
        "menu": 0,
        "title": 0,
        "other": 0,
    }
    assert result.blog_image_type_actuals == {
        "normal_gameplay": 0,
        "event": 3,
        "menu": 0,
        "title": 0,
        "other": 0,
    }
    assert "event_coverage" in result.selected[0].reason_codes
    assert all(
        "event_coverage" not in item.reason_codes for item in result.selected[1:]
    )


def test_available_ordinary_combat_and_event_each_receive_one_minimum_slot() -> None:
    """10枚以上の要求で通常戦闘とイベントが最低1枚ずつ選択されること。

    Arrange:
        - 低utilityだが有効な通常戦闘とイベントが各1件用意される
        - より高utilityな通常play候補が10件用意される
    Act:
        - 10枚のVideo Set選定が実行される
    Assert:
        - 通常戦闘とイベントが各1枚選択され、残り8枚が動的に選択されること
        - 条件付き最低coverageの候補数、最低数、実績、理由が記録されること
    """
    # Arrange
    feature_count = 12

    def unit_feature(index: int) -> tuple[float, ...]:
        return tuple(float(position == index) for position in range(feature_count))

    ordinary_combat = _candidate(
        "0",
        quality=0.1,
        feature=unit_feature(0),
        progress=Fraction(1, 100),
        blog_image_type="normal_gameplay",
        explanation_value="low",
        context_relevance="none",
        spoiler_risk="medium",
        combat_encounter_kind="ordinary",
    )
    event = _candidate(
        "1",
        quality=0.1,
        feature=unit_feature(1),
        progress=Fraction(2, 100),
        blog_image_type="event",
        explanation_value="low",
        context_relevance="none",
    )
    higher_utility = tuple(
        _candidate(
            digest,
            quality=0.9,
            feature=unit_feature(index),
            progress=Fraction(index + 1, 20),
            blog_image_type="normal_gameplay",
            explanation_value="high",
            context_relevance="none",
        )
        for index, digest in enumerate("23456789ab", start=2)
    )

    # Act
    result = select_video_set_images(
        (*higher_utility, event, ordinary_combat),
        requested_count=10,
        spoiler_sensitivity="medium",
        similarity_threshold=0.72,
    )

    # Assert
    selected_ids = {item.candidate.identifier for item in result.selected}
    assert len(result.selected) == 10
    assert ordinary_combat.identifier in selected_ids
    assert event.identifier in selected_ids
    assert result.selection_coverage_eligible_counts == {
        "ordinary_combat": 1,
        "event": 1,
    }
    assert result.selection_coverage_minimums == {
        "ordinary_combat": 1,
        "event": 1,
    }
    assert result.selection_coverage_actuals == {
        "ordinary_combat": 1,
        "event": 1,
    }
    assert result.selection_coverage_reallocated == {
        "ordinary_combat": False,
        "event": False,
    }
    reason_codes = {reason for item in result.selected for reason in item.reason_codes}
    assert "ordinary_combat_minimum_coverage" in reason_codes
    assert "event_minimum_coverage" in reason_codes


def test_missing_event_minimum_is_reallocated_without_invalid_event() -> None:
    """有効なイベントがない最低枠が他の有効候補へ再配分されること。

    Arrange:
        - 有効な通常戦闘1件と説明価値のないイベント1件が用意される
        - 十分な通常play候補が用意される
    Act:
        - 10枚のVideo Set選定が実行される
    Assert:
        - 通常戦闘が最低1枚選択され、無効イベントなしで10枚に到達すること
        - イベント最低数が0となり再配分済みとして記録されること
    """
    # Arrange
    feature_count = 12

    def unit_feature(index: int) -> tuple[float, ...]:
        return tuple(float(position == index) for position in range(feature_count))

    ordinary_combat = _candidate(
        "0",
        quality=0.1,
        feature=unit_feature(0),
        progress=Fraction(1, 100),
        blog_image_type="normal_gameplay",
        explanation_value="low",
        context_relevance="none",
        combat_encounter_kind="ordinary",
    )
    invalid_event = _candidate(
        "1",
        quality=1.0,
        feature=unit_feature(1),
        progress=Fraction(2, 100),
        blog_image_type="event",
        explanation_value="none",
        context_relevance="none",
    )
    other_candidates = tuple(
        _candidate(
            digest,
            quality=0.9,
            feature=unit_feature(index),
            progress=Fraction(index + 1, 20),
            blog_image_type="normal_gameplay",
            explanation_value="high",
            context_relevance="none",
        )
        for index, digest in enumerate("23456789ab", start=2)
    )

    # Act
    result = select_video_set_images(
        (*other_candidates, invalid_event, ordinary_combat),
        requested_count=10,
        spoiler_sensitivity="medium",
        similarity_threshold=0.72,
    )

    # Assert
    selected_ids = {item.candidate.identifier for item in result.selected}
    assert len(result.selected) == 10
    assert ordinary_combat.identifier in selected_ids
    assert invalid_event.identifier not in selected_ids
    assert result.selection_coverage_eligible_counts == {
        "ordinary_combat": 1,
        "event": 0,
    }
    assert result.selection_coverage_minimums == {
        "ordinary_combat": 1,
        "event": 0,
    }
    assert result.selection_coverage_actuals == {
        "ordinary_combat": 1,
        "event": 0,
    }
    assert result.selection_coverage_reallocated == {
        "ordinary_combat": False,
        "event": True,
    }


def test_impossible_minimum_restarts_unrestricted_selection_at_base_ceiling() -> None:
    """最低枠解放後の通常選定が設定済みsimilarity ceilingから再開されること。

    Arrange:
        - 通常戦闘と終端ceilingでも重複するイベント候補が用意される
        - base ceilingでは重複する高utility候補と、適格な代替候補9件が用意される
    Act:
        - 10枚のVideo Set選定が実行される
    Assert:
        - 不可能なイベント最低枠が解放され、base ceilingから通常選定されること
        - 高utilityな類似候補で適格な代替候補が押し出されないこと
    """
    # Arrange
    feature_count = 10

    def unit_feature(index: int) -> tuple[float, ...]:
        return tuple(float(position == index) for position in range(feature_count))

    ordinary_combat = _candidate(
        "0",
        quality=1.0,
        feature=unit_feature(0),
        progress=Fraction(1, 100),
        blog_image_type="normal_gameplay",
        explanation_value="high",
        context_relevance="none",
        combat_encounter_kind="ordinary",
    )
    impossible_event = _candidate(
        "1",
        quality=0.1,
        feature=unit_feature(0),
        progress=Fraction(2, 100),
        blog_image_type="event",
        explanation_value="low",
        context_relevance="none",
    )
    relaxed_only_candidate = _candidate(
        "2",
        quality=0.99,
        feature=(0.9, math.sqrt(1 - 0.9**2), *(0.0 for _ in range(8))),
        progress=Fraction(3, 100),
        blog_image_type="normal_gameplay",
        explanation_value="high",
        context_relevance="none",
    )
    base_eligible_candidates = tuple(
        _candidate(
            digest,
            quality=0.5,
            feature=unit_feature(index),
            progress=Fraction(index + 3, 100),
            blog_image_type="normal_gameplay",
            explanation_value="medium",
            context_relevance="none",
        )
        for index, digest in enumerate("3456789ab", start=1)
    )

    # Act
    result = select_video_set_images(
        (
            ordinary_combat,
            impossible_event,
            relaxed_only_candidate,
            *base_eligible_candidates,
        ),
        requested_count=10,
        spoiler_sensitivity="medium",
        similarity_threshold=0.72,
    )

    # Assert
    selected_ids = {item.candidate.identifier for item in result.selected}
    assert len(result.selected) == 10
    assert ordinary_combat.identifier in selected_ids
    assert impossible_event.identifier not in selected_ids
    assert relaxed_only_candidate.identifier not in selected_ids
    assert {candidate.identifier for candidate in base_eligible_candidates}.issubset(
        selected_ids
    )
    assert result.final_similarity_ceiling == 0.72
    assert result.selection_coverage_reallocated == {
        "ordinary_combat": False,
        "event": True,
    }
    relaxed_rejection = next(
        item
        for item in result.rejected
        if item.candidate.identifier == relaxed_only_candidate.identifier
    )
    assert relaxed_rejection.reason_code == "similarity_ceiling"
    assert relaxed_rejection.similarity == pytest.approx(0.9)


def test_satisfied_relaxed_minimum_restarts_unrestricted_selection_at_base() -> None:
    """緩和passで最低枠達成後の通常選定がbase ceilingへ戻ること。

    Arrange:
        - 通常戦闘と終端passだけで両立するイベントが用意される
        - 緩和時だけ選べる高utility候補とbase適格候補8件が用意される
    Act:
        - 10枚のVideo Set選定が実行される
    Assert:
        - イベント最低枠を保持してbase ceilingから通常選定が再開されること
        - 緩和時だけの候補よりbase適格候補8件が選択されること
    """
    # Arrange
    feature_count = 10

    def unit_feature(index: int) -> tuple[float, ...]:
        return tuple(float(position == index) for position in range(feature_count))

    ordinary_combat = _candidate(
        "0",
        quality=1.0,
        feature=unit_feature(0),
        progress=Fraction(1, 100),
        blog_image_type="normal_gameplay",
        explanation_value="high",
        context_relevance="none",
        combat_encounter_kind="ordinary",
    )
    event = _candidate(
        "1",
        quality=0.3,
        feature=(0.9, math.sqrt(1 - 0.9**2), *(0.0 for _ in range(8))),
        progress=Fraction(2, 100),
        blog_image_type="event",
        explanation_value="medium",
        context_relevance="none",
    )
    relaxed_only_candidate = _candidate(
        "2",
        quality=0.99,
        feature=(0.9, -math.sqrt(1 - 0.9**2), *(0.0 for _ in range(8))),
        progress=Fraction(3, 100),
        blog_image_type="normal_gameplay",
        explanation_value="high",
        context_relevance="none",
    )
    base_eligible_candidates = tuple(
        _candidate(
            digest,
            quality=0.5,
            feature=unit_feature(index),
            progress=Fraction(index + 3, 100),
            blog_image_type="normal_gameplay",
            explanation_value="medium",
            context_relevance="none",
        )
        for index, digest in enumerate("3456789a", start=2)
    )

    # Act
    result = select_video_set_images(
        (
            ordinary_combat,
            event,
            relaxed_only_candidate,
            *base_eligible_candidates,
        ),
        requested_count=10,
        spoiler_sensitivity="medium",
        similarity_threshold=0.72,
    )

    # Assert
    selected_ids = {item.candidate.identifier for item in result.selected}
    assert ordinary_combat.identifier in selected_ids
    assert event.identifier in selected_ids
    assert relaxed_only_candidate.identifier not in selected_ids
    assert {candidate.identifier for candidate in base_eligible_candidates}.issubset(
        selected_ids
    )
    assert result.final_similarity_ceiling == 0.72
    assert result.selection_coverage_reallocated == {
        "ordinary_combat": False,
        "event": False,
    }


def test_compatible_minimum_combination_is_preserved() -> None:
    """全facetを満たせる互換候補の組合せが高utility候補より優先されること。

    Arrange:
        - 唯一のイベントと終端ceilingで重複する高utility通常戦闘が用意される
        - イベントと互換性のある低utility通常戦闘と通常候補8件が用意される
    Act:
        - 10枚のVideo Set選定が実行される
    Assert:
        - 互換性のある通常戦闘とイベントが選択されること
        - 両方の最低coverageが再配分されないこと
    """
    # Arrange
    feature_count = 10

    def unit_feature(index: int) -> tuple[float, ...]:
        return tuple(float(position == index) for position in range(feature_count))

    incompatible_combat = _candidate(
        "0",
        quality=1.0,
        feature=unit_feature(0),
        progress=Fraction(1, 100),
        blog_image_type="normal_gameplay",
        explanation_value="high",
        context_relevance="none",
        combat_encounter_kind="ordinary",
    )
    event = _candidate(
        "1",
        quality=0.4,
        feature=unit_feature(0),
        progress=Fraction(2, 100),
        blog_image_type="event",
        explanation_value="medium",
        context_relevance="none",
    )
    compatible_combat = _candidate(
        "2",
        quality=0.8,
        feature=unit_feature(1),
        progress=Fraction(3, 100),
        blog_image_type="normal_gameplay",
        explanation_value="high",
        context_relevance="none",
        combat_encounter_kind="ordinary",
    )
    unrestricted = tuple(
        _candidate(
            digest,
            quality=0.7,
            feature=unit_feature(index),
            progress=Fraction(index + 3, 100),
            blog_image_type="normal_gameplay",
            explanation_value="high",
            context_relevance="none",
        )
        for index, digest in enumerate("3456789a", start=2)
    )

    # Act
    result = select_video_set_images(
        (incompatible_combat, event, compatible_combat, *unrestricted),
        requested_count=10,
        spoiler_sensitivity="medium",
        similarity_threshold=0.72,
    )

    # Assert
    selected_ids = {item.candidate.identifier for item in result.selected}
    assert compatible_combat.identifier in selected_ids
    assert event.identifier in selected_ids
    assert incompatible_combat.identifier not in selected_ids
    assert result.selection_coverage_actuals == {
        "ordinary_combat": 1,
        "event": 1,
    }
    assert result.selection_coverage_reallocated == {
        "ordinary_combat": False,
        "event": False,
    }


def test_joint_feasibility_reserves_variant_prerequisite_cost() -> None:
    """Variant Group前提が残り枠を超える最低枠の組合せが除外されること。

    Arrange:
        - 高utility通常戦闘とイベントが同じVariant Groupに属する
        - 同じsceneに未代表Groupが9件あり、別Groupの通常戦闘も用意される
    Act:
        - 10枚のVideo Set選定が実行される
    Assert:
        - 追加9枠を要する高utility通常戦闘ではなく別Group候補が選ばれること
        - 通常戦闘とイベントの最低枠が両方満たされること
    """
    # Arrange
    feature_count = 11

    def unit_feature(index: int) -> tuple[float, ...]:
        return tuple(float(position == index) for position in range(feature_count))

    expensive_combat = _candidate(
        "0",
        quality=1.0,
        feature=unit_feature(0),
        progress=Fraction(1, 100),
        blog_image_type="normal_gameplay",
        explanation_value="high",
        context_relevance="none",
        scene_selection_role="recurring_gameplay",
        scene_slug="battle",
        combat_encounter_kind="ordinary",
    )
    event = _candidate(
        "1",
        quality=0.8,
        feature=(0.96, math.sqrt(1 - 0.96**2), *(0.0 for _ in range(9))),
        progress=Fraction(2, 100),
        blog_image_type="event",
        explanation_value="high",
        context_relevance="none",
        scene_selection_role="recurring_gameplay",
        scene_slug="battle",
    )
    compatible_combat = _candidate(
        "2",
        quality=0.7,
        feature=unit_feature(2),
        progress=Fraction(3, 100),
        blog_image_type="normal_gameplay",
        explanation_value="high",
        context_relevance="none",
        scene_selection_role="recurring_gameplay",
        scene_slug="battle",
        combat_encounter_kind="ordinary",
    )
    other_groups = tuple(
        _candidate(
            digest,
            quality=0.6,
            feature=unit_feature(index),
            progress=Fraction(index + 3, 100),
            blog_image_type="normal_gameplay",
            explanation_value="high",
            context_relevance="none",
            scene_selection_role="recurring_gameplay",
            scene_slug="battle",
        )
        for index, digest in enumerate("3456789a", start=3)
    )

    # Act
    result = select_video_set_images(
        (expensive_combat, event, compatible_combat, *other_groups),
        requested_count=10,
        spoiler_sensitivity="medium",
        similarity_threshold=0.72,
    )

    # Assert
    selected_ids = {item.candidate.identifier for item in result.selected}
    assert compatible_combat.identifier in selected_ids
    assert event.identifier in selected_ids
    assert expensive_combat.identifier not in selected_ids
    assert result.selection_coverage_actuals == {
        "ordinary_combat": 1,
        "event": 1,
    }
    assert result.selection_coverage_reallocated == {
        "ordinary_combat": False,
        "event": False,
    }


def test_joint_feasibility_collapses_title_limited_variant_prerequisites() -> None:
    """Title上限で無効になるVariant Groupが前提枠へ重複計上されないこと。

    Arrange:
        - 高utility通常戦闘とイベントが同じVariant Groupに属する
        - 同じsceneにtitleだけを持つ未代表Groupが9件用意される
        - 別sceneに低utility通常戦闘と通常候補7件が用意される
    Act:
        - 10枚のVideo Set選定が実行される
    Assert:
        - title 1枚で残りtitle Groupが無効になり高utilityの最低枠候補が選ばれること
        - 低utility通常戦闘に最低枠が置き換えられないこと
    """
    # Arrange
    candidates = _hard_limit_variant_candidates("title")
    repeated_combat, event, alternative_combat = candidates[:3]

    # Act
    result = select_video_set_images(
        candidates,
        requested_count=10,
        spoiler_sensitivity="medium",
        similarity_threshold=0.72,
    )

    # Assert
    selected_ids = {item.candidate.identifier for item in result.selected}
    repeated_selection = next(
        item
        for item in result.selected
        if item.candidate.identifier == repeated_combat.identifier
    )
    assert repeated_combat.identifier in selected_ids
    assert event.identifier in selected_ids
    assert alternative_combat.identifier not in selected_ids
    assert "ordinary_combat_minimum_coverage" in repeated_selection.reason_codes
    assert result.blog_image_type_actuals["title"] == 1


def test_joint_feasibility_collapses_spoiler_limited_variant_prerequisites() -> None:
    """Spoiler上限で無効になるVariant Groupが前提枠へ重複計上されないこと。

    Arrange:
        - 高utility通常戦闘とイベントが同じVariant Groupに属する
        - 同じsceneにMajor Spoilerだけを持つ未代表Groupが9件用意される
        - low感度の選定でMajor Spoiler上限が1枚になる候補順が用意される
    Act:
        - medium感度で10枚のVideo Set選定が実行される
    Assert:
        - Major Spoiler 1枚で残りGroupが無効になり高utilityの最低枠候補が選ばれること
        - 低utility通常戦闘に最低枠が置き換えられないこと
    """
    # Arrange
    candidates = _hard_limit_variant_candidates("spoiler")
    repeated_combat, event, alternative_combat = candidates[:3]

    # Act
    result = select_video_set_images(
        candidates,
        requested_count=10,
        spoiler_sensitivity="medium",
        similarity_threshold=0.72,
    )

    # Assert
    selected_ids = {item.candidate.identifier for item in result.selected}
    repeated_selection = next(
        item
        for item in result.selected
        if item.candidate.identifier == repeated_combat.identifier
    )
    assert repeated_combat.identifier in selected_ids
    assert event.identifier in selected_ids
    assert alternative_combat.identifier not in selected_ids
    assert "ordinary_combat_minimum_coverage" in repeated_selection.reason_codes
    assert result.major_spoiler_limit == 1
    assert result.major_spoiler_selected_count == 1


def test_prerequisites_preserve_a_completable_coverage_path() -> None:
    """残り枠へ収まる最低coverage経路の前提候補だけが保持されること。

    Arrange:
        - 選択済みGroupを再利用するイベント経路が二つ用意される
        - 一方は高utilityの未代表Group 8件、他方は低utilityの1件を必要とする
        - 前提とイベントに使える残り枠が2件だけ用意される
    Act:
        - 選択可能な前提候補から完了可能な経路が抽出される
    Assert:
        - 2枠へ収まる低utility前提だけが保持されること
        - 収まらない経路の高utility前提が混入しないこと
    """
    # Arrange
    feature_count = 10

    def unit_feature(index: int) -> tuple[float, ...]:
        return tuple(float(position == index) for position in range(feature_count))

    selected_candidates = tuple(
        _candidate(
            f"selected-{scene}",
            quality=0.8,
            feature=unit_feature(0),
            progress=Fraction(index + 1, 100),
            blog_image_type="normal_gameplay",
            explanation_value="high",
            context_relevance="none",
            scene_selection_role="recurring_gameplay",
            scene_slug=scene,
        )
        for index, scene in enumerate(("infeasible", "feasible"))
    )
    selected = [
        select_video_set_images(
            (candidate,),
            requested_count=1,
            spoiler_sensitivity="medium",
            similarity_threshold=0.72,
        ).selected[0]
        for candidate in selected_candidates
    ]
    events = tuple(
        _candidate(
            f"event-{scene}",
            quality=0.8,
            feature=(0.96, math.sqrt(1 - 0.96**2), *(0.0 for _ in range(8))),
            progress=Fraction(index + 3, 100),
            blog_image_type="event",
            explanation_value="high",
            context_relevance="none",
            scene_selection_role="recurring_gameplay",
            scene_slug=scene,
        )
        for index, scene in enumerate(("infeasible", "feasible"))
    )
    infeasible_prerequisites = tuple(
        _candidate(
            f"infeasible-prerequisite-{index}",
            quality=1.0,
            feature=unit_feature(index),
            progress=Fraction(index + 5, 100),
            blog_image_type="normal_gameplay",
            explanation_value="high",
            context_relevance="none",
            scene_selection_role="recurring_gameplay",
            scene_slug="infeasible",
        )
        for index in range(2, 10)
    )
    feasible_prerequisite = _candidate(
        "feasible-prerequisite",
        quality=0.1,
        feature=unit_feature(2),
        progress=Fraction(15, 100),
        blog_image_type="normal_gameplay",
        explanation_value="low",
        context_relevance="none",
        scene_selection_role="recurring_gameplay",
        scene_slug="feasible",
    )
    prerequisites = (*infeasible_prerequisites, feasible_prerequisite)
    evaluated = [
        (
            candidate,
            select_video_set_images(
                (candidate,),
                requested_count=1,
                spoiler_sensitivity="medium",
                similarity_threshold=0.72,
            )
            .selected[0]
            .score,
        )
        for candidate in prerequisites
    ]
    variant_groups = {
        events[0].identifier: selected[0].variant_group_id,
        events[1].identifier: selected[1].variant_group_id,
        **{
            candidate.identifier: f"variant_infeasible_{index}"
            for index, candidate in enumerate(infeasible_prerequisites)
        },
        feasible_prerequisite.identifier: "variant_feasible_prerequisite",
    }
    unmet_facets: set[SelectionCoverageFacet] = {"event"}

    # Act
    result = select_completable_coverage_prerequisites(
        evaluated,
        [*events, *prerequisites],
        selected,
        {"title": 0},
        unmet_facets,
        variant_groups,
        {},
        0.97,
        None,
        2,
    )

    # Assert
    assert [candidate.identifier for candidate, _score in result] == [
        feasible_prerequisite.identifier
    ]


def test_variant_prerequisite_advances_while_minimum_slot_is_reserved() -> None:
    """未代表Variant Groupが先行し実現可能な最低coverageが保持されること。

    Arrange:
        - 同一recurring sceneの通常戦闘とイベントが同じVariant Groupに属する
        - イベント選択前に別Variant Groupの低utility候補を代表させる必要がある
        - 通常候補だけでも要求10枚を満たせる候補集合が用意される
    Act:
        - 10枚のVideo Set選定が実行される
    Assert:
        - prerequisite候補の枠とイベント最低枠が予約されること
        - prerequisite、イベントの順に選択され最低coverageが解放されないこと
    """
    # Arrange
    feature_count = 11

    def unit_feature(index: int) -> tuple[float, ...]:
        return tuple(float(position == index) for position in range(feature_count))

    ordinary_combat = _candidate(
        "0",
        quality=1.0,
        feature=unit_feature(0),
        progress=Fraction(1, 100),
        blog_image_type="normal_gameplay",
        explanation_value="high",
        context_relevance="none",
        scene_selection_role="recurring_gameplay",
        scene_slug="battle",
        combat_encounter_kind="ordinary",
    )
    event = _candidate(
        "1",
        quality=0.9,
        feature=(0.96, math.sqrt(1 - 0.96**2), *(0.0 for _ in range(9))),
        progress=Fraction(2, 100),
        blog_image_type="event",
        explanation_value="high",
        context_relevance="none",
        scene_selection_role="recurring_gameplay",
        scene_slug="battle",
    )
    prerequisite = _candidate(
        "2",
        quality=0.1,
        feature=unit_feature(2),
        progress=Fraction(3, 100),
        blog_image_type="normal_gameplay",
        explanation_value="low",
        context_relevance="none",
        scene_selection_role="recurring_gameplay",
        scene_slug="battle",
    )
    unrestricted = tuple(
        _candidate(
            digest,
            quality=0.8,
            feature=unit_feature(index),
            progress=Fraction(index + 3, 100),
            blog_image_type="normal_gameplay",
            explanation_value="high",
            context_relevance="none",
        )
        for index, digest in enumerate("3456789a", start=3)
    )

    # Act
    result = select_video_set_images(
        (ordinary_combat, event, prerequisite, *unrestricted),
        requested_count=10,
        spoiler_sensitivity="medium",
        similarity_threshold=0.72,
    )

    # Assert
    selected_ids = [item.candidate.identifier for item in result.selected]
    assert selected_ids[:3] == [
        ordinary_combat.identifier,
        prerequisite.identifier,
        event.identifier,
    ]
    assert len(selected_ids) == 10
    assert result.selection_coverage_actuals == {
        "ordinary_combat": 1,
        "event": 1,
    }
    assert result.selection_coverage_reallocated == {
        "ordinary_combat": False,
        "event": False,
    }


def test_higher_spoiler_sensitivity_never_increases_major_spoilers() -> None:
    """感度上昇で同じ候補集合のMajor Spoiler選択数が増えないこと。

    Arrange:
        - 単純greedyではdiversityとcoverageによりhigh感度の件数が増える候補行列がある
        - 同じ7候補と要求3枚がlow、medium、high向けに用意される
    Act:
        - 各Spoiler SensitivityでVideo Set selectorが実行される
    Assert:
        - Major Spoiler選択数が感度順に単調非増加であること
    """
    # Arrange
    specs: tuple[CandidateSpec, ...] = (
        (
            "0",
            0.38,
            (0.3010184307, 0.2251490036, 0.8531105982, 0.3617984767),
            Fraction(18, 25),
            "event",
            "low",
            "high",
        ),
        (
            "1",
            0.70,
            (0.9405985621, 0.1960576527, 0.2755141423, 0.0304581544),
            Fraction(17, 25),
            "event",
            "low",
            "none",
        ),
        (
            "2",
            0.53,
            (0.6382307865, 0.1248780060, 0.7098586663, 0.2704951396),
            Fraction(69, 100),
            "normal_gameplay",
            "none",
            "high",
        ),
        (
            "3",
            0.90,
            (0.5763720210, 0.1029043598, 0.6540056522, 0.4790434145),
            Fraction(47, 50),
            "normal_gameplay",
            "high",
            "high",
        ),
        (
            "4",
            0.81,
            (0.8368746524, 0.1520583373, 0.2970922673, 0.4338839281),
            Fraction(7, 100),
            "event",
            "high",
            "none",
        ),
        (
            "5",
            0.48,
            (0.8195546317, 0.3893717013, 0.4171090861, 0.0523439990),
            Fraction(91, 100),
            "event",
            "low",
            "high",
        ),
        (
            "6",
            0.44,
            (0.6511259894, 0.2176397414, 0.7058702682, 0.1743991209),
            Fraction(3, 50),
            "menu",
            "high",
            "high",
        ),
    )
    candidates = tuple(
        _candidate(
            digest,
            quality=quality,
            feature=feature,
            progress=progress,
            blog_image_type=image_type,
            explanation_value=explanation,
            context_relevance="none",
            spoiler_risk=spoiler,
        )
        for (
            digest,
            quality,
            feature,
            progress,
            image_type,
            explanation,
            spoiler,
        ) in specs
    )

    # Act
    sensitivities: tuple[SpoilerSensitivity, ...] = ("low", "medium", "high")
    results = tuple(
        select_video_set_images(
            candidates,
            requested_count=3,
            spoiler_sensitivity=sensitivity,
            similarity_threshold=0.72,
        )
        for sensitivity in sensitivities
    )

    # Assert
    major_counts = [result.major_spoiler_selected_count for result in results]
    assert major_counts[0] == 1
    assert major_counts == sorted(major_counts, reverse=True)


def test_shortlist_expansion_recomputes_selection_from_full_expanded_pool() -> None:
    """Shortlist拡張後に以前の選択を固定せず全候補から再計算されること。

    Arrange:
        - 初期batchには要求数を満たさない低utility候補が1件だけある
        - 次のbatchには視覚的に異なる高utility候補が2件ある
    Act:
        - lazyなSelection Shortlist batch列から選定される
    Assert:
        - 拡張poolの高utility候補2件が選ばれ初期候補が固定されないこと
        - annotation件数と拡張回数が返されること
    """
    # Arrange
    initial = _candidate(
        "9",
        quality=0.5,
        feature=(1.0, 0.0, 0.0),
        progress=Fraction(1, 10),
        blog_image_type="normal_gameplay",
        explanation_value="none",
        context_relevance="none",
    )
    event = _candidate(
        "a",
        quality=0.9,
        feature=(0.0, 1.0, 0.0),
        progress=Fraction(5, 10),
        blog_image_type="event",
        explanation_value="high",
        context_relevance="none",
    )
    gameplay = _candidate(
        "b",
        quality=0.8,
        feature=(0.0, 0.0, 1.0),
        progress=Fraction(9, 10),
        blog_image_type="normal_gameplay",
        explanation_value="high",
        context_relevance="none",
    )

    # Act
    result = select_from_shortlist_batches(
        ((initial,), (gameplay, event)),
        candidate_moment_timelines=_candidate_moment_timelines(
            (initial, gameplay, event)
        ),
        requested_count=2,
        spoiler_sensitivity="medium",
        similarity_threshold=0.72,
    )

    # Assert
    assert [item.candidate.identifier for item in result.selected] == [
        event.identifier,
        gameplay.identifier,
    ]
    assert [item.candidate.identifier for item in result.rejected] == [
        initial.identifier
    ]
    assert result.rejected[0].reason_code == "lower_marginal_utility"
    assert result.annotated_candidate_count == 3
    assert result.shortlist_expansion_count == 1
    assert result.all_candidate_moments_exhausted is False


def test_shortlist_expands_for_an_undiscovered_conditional_facet() -> None:
    """要求枚数到達後も未発見の条件付きfacetを探してbatchが拡張されること。

    Arrange:
        - 初期batchに通常戦闘1件を含む選択可能な通常画像10件が用意される
        - 次のbatchに有効なイベント1件が用意される
    Act:
        - lazyなSelection Shortlist batch列から10枚が選定される
    Assert:
        - 次batchまで拡張されイベント最低枠が満たされること
        - 通常戦闘とイベントの最低枠が再配分されないこと
    """
    # Arrange
    feature_count = 11

    def unit_feature(index: int) -> tuple[float, ...]:
        return tuple(float(position == index) for position in range(feature_count))

    ordinary_combat = _candidate(
        "0",
        quality=0.9,
        feature=unit_feature(0),
        progress=Fraction(1, 100),
        blog_image_type="normal_gameplay",
        explanation_value="high",
        context_relevance="none",
        combat_encounter_kind="ordinary",
    )
    initial_gameplay = tuple(
        _candidate(
            digest,
            quality=0.8,
            feature=unit_feature(index),
            progress=Fraction(index + 1, 20),
            blog_image_type="normal_gameplay",
            explanation_value="high",
            context_relevance="none",
        )
        for index, digest in enumerate("123456789", start=1)
    )
    event = _candidate(
        "a",
        quality=0.1,
        feature=unit_feature(10),
        progress=Fraction(9, 10),
        blog_image_type="event",
        explanation_value="low",
        context_relevance="none",
    )

    # Act
    result = select_from_shortlist_batches(
        ((ordinary_combat, *initial_gameplay), (event,)),
        candidate_moment_timelines=_candidate_moment_timelines(
            (ordinary_combat, *initial_gameplay, event)
        ),
        requested_count=10,
        spoiler_sensitivity="medium",
        similarity_threshold=0.72,
    )

    # Assert
    selected_ids = {item.candidate.identifier for item in result.selected}
    assert event.identifier in selected_ids
    assert result.shortlist_expansion_count == 1
    assert result.selection_coverage_actuals == {
        "ordinary_combat": 1,
        "event": 1,
    }
    assert result.selection_coverage_reallocated == {
        "ordinary_combat": False,
        "event": False,
    }


def test_shortlist_expands_until_combat_encounter_boundary_is_observed() -> None:
    """未注釈Momentが主要戦闘Group内に残る間は選定が確定されないこと。

    Arrange:
        - 同じsource・Scene Slugの主要戦闘2件が初期batchに用意される
        - 2件の間にある非戦闘Momentだけが後続batchに用意される
        - 初期batchだけでも要求枚数は満たせる
    Act:
        - 完全なCandidate Moment時系列付きでlazy batch選定が実行される
    Assert:
        - 後続batchまで注釈され別遭遇の主要戦闘2件が選択されること
    """
    # Arrange
    source_fingerprint = "a" * 64
    first_encounter = _candidate(
        "hidden-boundary-first",
        quality=0.99,
        feature=(1.0, 0.0, 0.0, 0.0),
        progress=Fraction(10, 100),
        blog_image_type="normal_gameplay",
        explanation_value="high",
        context_relevance="none",
        scene_slug="repeated-boss",
        combat_encounter_kind="major",
        video_fingerprint=source_fingerprint,
    )
    boundary = _candidate(
        "hidden-boundary-event",
        quality=0.10,
        feature=(0.0, 1.0, 0.0, 0.0),
        progress=Fraction(20, 100),
        blog_image_type="event",
        explanation_value="none",
        context_relevance="none",
        scene_slug="story-event",
        video_fingerprint=source_fingerprint,
    )
    second_encounter = _candidate(
        "hidden-boundary-second",
        quality=0.98,
        feature=(0.0, 0.0, 1.0, 0.0),
        progress=Fraction(30, 100),
        blog_image_type="normal_gameplay",
        explanation_value="high",
        context_relevance="none",
        scene_slug="repeated-boss",
        combat_encounter_kind="major",
        video_fingerprint=source_fingerprint,
    )
    filler = _candidate(
        "hidden-boundary-filler",
        quality=0.10,
        feature=(0.0, 0.0, 0.0, 1.0),
        progress=Fraction(80, 100),
        blog_image_type="normal_gameplay",
        explanation_value="high",
        context_relevance="none",
        scene_slug="exploration",
        video_fingerprint=source_fingerprint,
    )
    all_candidates = (first_encounter, boundary, second_encounter, filler)

    # Act
    result = select_from_shortlist_batches(
        ((first_encounter, second_encounter, filler), (boundary,)),
        candidate_moment_timelines=_candidate_moment_timelines(all_candidates),
        requested_count=2,
        spoiler_sensitivity="medium",
        similarity_threshold=0.72,
    )

    # Assert
    assert {item.candidate.identifier for item in result.selected} == {
        first_encounter.identifier,
        second_encounter.identifier,
    }
    assert result.annotated_candidate_count == 4
    assert result.shortlist_expansion_count == 1


def test_title_semantics_do_not_complete_event_minimum_before_real_event() -> None:
    """eventへ誤分類されたタイトルでevent最低枠が充足されないこと。

    Arrange:
        - 通常戦闘、eventへ誤分類されたタイトル、通常画像8件が初期batchにある
        - 実際のevent候補が後続batchにある
    Act:
        - 10枚のlazy batch選定が実行される
    Assert:
        - 後続batchまで注釈され実際のeventが最低枠として選択されること
        - タイトルがevent最低枠の実績へ数えられないこと
    """
    # Arrange
    feature_count = 11

    def unit_feature(index: int) -> tuple[float, ...]:
        return tuple(float(position == index) for position in range(feature_count))

    ordinary_combat = _candidate(
        "title-facet-combat",
        quality=0.95,
        feature=unit_feature(0),
        progress=Fraction(1, 100),
        blog_image_type="normal_gameplay",
        explanation_value="high",
        context_relevance="none",
        combat_encounter_kind="ordinary",
    )
    misclassified_title = _candidate(
        "title-facet-misclassified",
        quality=0.94,
        feature=unit_feature(1),
        progress=Fraction(2, 100),
        blog_image_type="event",
        explanation_value="high",
        context_relevance="none",
        screen_text_kind="title",
        content_kind="title",
    )
    gameplay = tuple(
        _candidate(
            f"title-facet-gameplay-{index}",
            quality=0.80 - index / 100,
            feature=unit_feature(index + 2),
            progress=Fraction(index + 2, 20),
            blog_image_type="normal_gameplay",
            explanation_value="high",
            context_relevance="none",
        )
        for index in range(8)
    )
    event = _candidate(
        "title-facet-real-event",
        quality=0.90,
        feature=unit_feature(10),
        progress=Fraction(90, 100),
        blog_image_type="event",
        explanation_value="high",
        context_relevance="none",
    )
    all_candidates = (ordinary_combat, misclassified_title, *gameplay, event)

    # Act
    result = select_from_shortlist_batches(
        ((ordinary_combat, misclassified_title, *gameplay), (event,)),
        candidate_moment_timelines=_candidate_moment_timelines(all_candidates),
        requested_count=10,
        spoiler_sensitivity="medium",
        similarity_threshold=0.72,
    )

    # Assert
    selected_ids = {item.candidate.identifier for item in result.selected}
    assert event.identifier in selected_ids
    assert result.shortlist_expansion_count == 1
    assert result.selection_coverage_actuals["event"] == 1


def test_exact_utility_tie_uses_stable_video_order_and_records_tie_break() -> None:
    """完全同点がVideo Orderで安定解消され診断へ残ること。

    Arrange:
        - utility、spoiler、quality、選択前visual similarityが同じ2候補がある
        - 入力順の後ろにある候補のVideo Orderが小さい
    Act:
        - 1枚の選定が要求される
    Assert:
        - 小さいVideo Orderの候補が選ばれること
        - stable tie-breakが使われたことをreasonとfieldで確認できること
    """
    # Arrange
    later_video = _candidate(
        "c",
        quality=0.8,
        feature=(1.0, 0.0),
        progress=Fraction(1, 10),
        blog_image_type="normal_gameplay",
        explanation_value="high",
        context_relevance="none",
        video_order=1,
    )
    earlier_video = _candidate(
        "d",
        quality=0.8,
        feature=(0.0, 1.0),
        progress=Fraction(9, 10),
        blog_image_type="normal_gameplay",
        explanation_value="high",
        context_relevance="none",
        video_order=0,
    )

    # Act
    result = select_video_set_images(
        (later_video, earlier_video),
        requested_count=1,
        spoiler_sensitivity="medium",
        similarity_threshold=0.72,
    )

    # Assert
    selected = result.selected[0]
    assert selected.candidate.identifier == earlier_video.identifier
    assert selected.tie_break_applied is True
    assert "stable_tie_break" in selected.reason_codes


def test_similarity_above_terminal_ceiling_is_counted_separately() -> None:
    """0.97超の候補がVisual Near-Duplicateとは別理由で集計されること。

    Arrange:
        - cosine similarityが0.97超0.995以下のordinary候補が2件ある
        - 2枚の選定が要求される
    Act:
        - selectorが終端similarity passまで実行される
    Assert:
        - 2件目がsimilarity ceilingで拒否されること
        - shortfallのreason countをstable enumで説明できること
    """
    # Arrange
    first = _candidate(
        "e",
        quality=0.9,
        feature=(1.0, 0.0),
        progress=Fraction(1, 10),
        blog_image_type="normal_gameplay",
        explanation_value="high",
        context_relevance="none",
    )
    second = _candidate(
        "f",
        quality=0.8,
        feature=(0.985, math.sqrt(1 - 0.985**2)),
        progress=Fraction(9, 10),
        blog_image_type="normal_gameplay",
        explanation_value="high",
        context_relevance="none",
    )

    # Act
    result = select_video_set_images(
        (second, first),
        requested_count=2,
        spoiler_sensitivity="medium",
        similarity_threshold=0.72,
    )

    # Assert
    assert result.shortfall is True
    assert result.final_similarity_ceiling == 0.97
    assert result.rejection_counts == {"similarity_ceiling": 1}
    rejection = result.rejected[0]
    assert rejection.reason_code == "similarity_ceiling"
    assert rejection.nearest_selected_image_id == first.identifier
    assert rejection.similarity == pytest.approx(0.985)


def test_rejection_uses_pass_that_satisfied_request() -> None:
    """要求数を満たした実際のpassでsimilarity rejectionが説明されること。

    Arrange:
        - base passで選ばれる2候補とbase超0.97以下の高utility候補がある
    Act:
        - base passで2枚の選定が満たされる
    Assert:
        - 未採用候補が最終到達passによるsimilarity ceilingとして返されること
    """
    # Arrange
    first = _candidate(
        "e",
        quality=0.9,
        feature=(1.0, 0.0),
        progress=Fraction(1, 10),
        blog_image_type="normal_gameplay",
        explanation_value="high",
        context_relevance="none",
    )
    similar_near_miss = _candidate(
        "f",
        quality=0.89,
        feature=(0.9, math.sqrt(1 - 0.9**2)),
        progress=Fraction(9, 10),
        blog_image_type="normal_gameplay",
        explanation_value="high",
        context_relevance="none",
    )
    diverse_filler = _candidate(
        "0",
        quality=0.5,
        feature=(0.0, 1.0),
        progress=Fraction(9, 10),
        blog_image_type="normal_gameplay",
        explanation_value="low",
        context_relevance="none",
    )

    # Act
    result = select_video_set_images(
        (similar_near_miss, diverse_filler, first),
        requested_count=2,
        spoiler_sensitivity="medium",
        similarity_threshold=0.72,
    )

    # Assert
    assert result.final_similarity_ceiling == 0.72
    rejection = result.rejected[0]
    assert rejection.candidate.identifier == similar_near_miss.identifier
    assert rejection.reason_code == "similarity_ceiling"
    assert rejection.nearest_selected_image_id == first.identifier
    assert rejection.similarity == pytest.approx(0.9)


def test_rejections_are_ordered_by_counterfactual_utility() -> None:
    """near-miss候補が反実仮想utilityとstable tie-break順で返されること。

    Arrange:
        - 選択1件と、ID順がutility順の逆になる未採用2件が用意される
    Act:
        - 1枚の選定が要求される
    Assert:
        - 未採用候補がID順でなくcounterfactual utility降順になること
    """
    # Arrange
    selected = _candidate(
        "9",
        quality=0.9,
        feature=(1.0, 0.0, 0.0),
        progress=Fraction(1, 10),
        blog_image_type="normal_gameplay",
        explanation_value="low",
        context_relevance="none",
    )
    stronger_near_miss = _candidate(
        "f",
        quality=0.8,
        feature=(0.0, 1.0, 0.0),
        progress=Fraction(5, 10),
        blog_image_type="normal_gameplay",
        explanation_value="low",
        context_relevance="none",
    )
    weaker_near_miss = _candidate(
        "0",
        quality=0.1,
        feature=(0.0, 0.0, 1.0),
        progress=Fraction(9, 10),
        blog_image_type="normal_gameplay",
        explanation_value="low",
        context_relevance="none",
    )

    # Act
    result = select_video_set_images(
        (weaker_near_miss, selected, stronger_near_miss),
        requested_count=1,
        spoiler_sensitivity="medium",
        similarity_threshold=0.72,
    )

    # Assert
    assert [item.candidate.identifier for item in result.rejected] == [
        stronger_near_miss.identifier,
        weaker_near_miss.identifier,
    ]
    assert [
        round(item.counterfactual_score.marginal_utility, 2) for item in result.rejected
    ] == [0.74, 0.25]


def test_same_major_combat_encounter_selects_only_the_best_representative() -> None:
    """分類揺れを含む同一主要戦闘から最良の1枚だけが選択されること。

    Arrange:
        - 同じ主要戦闘Aの候補3件に一時的なScene Slug誤分類が含まれる
        - 別の主要戦闘Bと、その後に始まる同名の別遭遇Aが用意される
        - 同じ候補集合が異なる入力順で用意される
    Act:
        - 5枚の選定が両方の候補集合へ要求される
    Assert:
        - 最初の戦闘Aから最高utilityの1枚だけが選択されること
        - 戦闘Bと後発の別遭遇Aが維持されること
        - 除外候補へstable reasonとblocking selected IDが記録されること
    """
    # Arrange
    source_fingerprint = "1" * 64
    encounter_a_best = _candidate(
        "encounter-a-best",
        quality=0.95,
        feature=(1.0, 0.0, 0.0, 0.0, 0.0),
        progress=Fraction(10, 100),
        blog_image_type="normal_gameplay",
        explanation_value="high",
        context_relevance="none",
        scene_selection_role="recurring_gameplay",
        scene_slug="boss-a",
        combat_encounter_kind="major",
        video_fingerprint=source_fingerprint,
    )
    encounter_a_misclassified = _candidate(
        "encounter-a-misclassified",
        quality=0.90,
        feature=(0.0, 1.0, 0.0, 0.0, 0.0),
        progress=Fraction(12, 100),
        blog_image_type="normal_gameplay",
        explanation_value="high",
        context_relevance="none",
        scene_selection_role="recurring_gameplay",
        scene_slug="mistaken-boss-name",
        combat_encounter_kind="major",
        video_fingerprint=source_fingerprint,
    )
    encounter_a_other_variant = _candidate(
        "encounter-a-other-variant",
        quality=0.85,
        feature=(0.0, 0.0, 1.0, 0.0, 0.0),
        progress=Fraction(14, 100),
        blog_image_type="normal_gameplay",
        explanation_value="high",
        context_relevance="none",
        scene_selection_role="recurring_gameplay",
        scene_slug="boss-a",
        combat_encounter_kind="major",
        video_fingerprint=source_fingerprint,
    )
    encounter_b = _candidate(
        "encounter-b",
        quality=0.80,
        feature=(0.0, 0.0, 0.0, 1.0, 0.0),
        progress=Fraction(40, 100),
        blog_image_type="normal_gameplay",
        explanation_value="high",
        context_relevance="none",
        scene_selection_role="recurring_gameplay",
        scene_slug="boss-b",
        combat_encounter_kind="major",
        video_fingerprint=source_fingerprint,
    )
    later_encounter_a = _candidate(
        "later-encounter-a",
        quality=0.75,
        feature=(0.0, 0.0, 0.0, 0.0, 1.0),
        progress=Fraction(70, 100),
        blog_image_type="normal_gameplay",
        explanation_value="high",
        context_relevance="none",
        scene_selection_role="recurring_gameplay",
        scene_slug="boss-a",
        combat_encounter_kind="major",
        video_fingerprint=source_fingerprint,
    )
    candidates = (
        encounter_a_best,
        encounter_a_misclassified,
        encounter_a_other_variant,
        encounter_b,
        later_encounter_a,
    )

    # Act
    forward = select_video_set_images(
        candidates,
        requested_count=5,
        spoiler_sensitivity="medium",
        similarity_threshold=0.72,
    )
    reversed_result = select_video_set_images(
        tuple(reversed(candidates)),
        requested_count=5,
        spoiler_sensitivity="medium",
        similarity_threshold=0.72,
    )

    # Assert
    expected_selected_ids = {
        encounter_a_best.identifier,
        encounter_b.identifier,
        later_encounter_a.identifier,
    }
    assert {item.candidate.identifier for item in forward.selected} == (
        expected_selected_ids
    )
    assert [item.candidate.identifier for item in forward.selected] == [
        item.candidate.identifier for item in reversed_result.selected
    ]
    expected_rejected_ids = {
        encounter_a_misclassified.identifier,
        encounter_a_other_variant.identifier,
    }
    assert {item.candidate.identifier for item in forward.rejected} == (
        expected_rejected_ids
    )
    for rejection in forward.rejected:
        assert rejection.reason_code == "semantic_duplicate"
        assert rejection.blocked_by_image_id == encounter_a_best.identifier
        assert rejection.semantic_group_id == (
            next(
                item.semantic_group_id
                for item in forward.selected
                if item.candidate.identifier == encounter_a_best.identifier
            )
        )
        assert rejection.semantic_group_basis == "combat_encounter_sequence"
    encounter_representative = next(
        item
        for item in forward.selected
        if item.candidate.identifier == encounter_a_best.identifier
    )
    assert "semantic_group_representative" in encounter_representative.reason_codes
    assert encounter_representative.semantic_group_id is not None
    assert encounter_representative.semantic_group_basis == "combat_encounter_sequence"


def test_same_subject_across_videos_selects_best_representative() -> None:
    """名前と動画が異なる同じ戦闘対象から最良の1枚だけが選択されること。

    Arrange:
        - 同じ外見の主要戦闘対象が異なるVideo Sourceと誤った名前で注釈される
        - 外見の異なる主要戦闘対象が用意される
    Act:
        - 全候補数と同じ3枚の選定が要求される
    Assert:
        - 同じ戦闘対象から最高utilityの1枚だけが選択されること
        - 外見の異なる戦闘対象が維持されること
        - 除外候補へCombat Subject Groupの根拠と代表IDが記録されること
    """
    # Arrange
    same_subject = CombatSubjectEvidence(
        body_plan="quadruped",
        scale="large",
        surface="organic",
        colors=("brown", "green"),
        traits=("bulbous_body", "large_mouth"),
        distinctiveness="distinctive",
    )
    different_subject = CombatSubjectEvidence(
        body_plan="humanoid",
        scale="large",
        surface="armored",
        colors=("black", "red"),
        traits=("armor", "weapon"),
        distinctiveness="distinctive",
    )
    best = _candidate(
        "same-subject-best",
        quality=0.95,
        feature=(1.0, 0.0, 0.0),
        progress=Fraction(10, 100),
        blog_image_type="normal_gameplay",
        explanation_value="high",
        context_relevance="none",
        scene_selection_role="recurring_gameplay",
        scene_slug="progyo-no-oyabun",
        combat_encounter_kind="major",
        video_order=0,
        video_fingerprint="a" * 64,
        summary="プロギョの大親分との戦闘",
        combat_subject_evidence=same_subject,
    )
    weaker = _candidate(
        "same-subject-weaker",
        quality=0.80,
        feature=(0.90, math.sqrt(1 - 0.90**2), 0.0),
        progress=Fraction(55, 100),
        blog_image_type="other",
        explanation_value="high",
        context_relevance="none",
        scene_selection_role="ordinary",
        scene_slug="frogi-no-oyabun",
        combat_encounter_kind="major",
        video_order=1,
        video_fingerprint="b" * 64,
        summary="フロギーの大親分との戦闘",
        combat_subject_evidence=same_subject,
    )
    other_boss = _candidate(
        "different-subject",
        quality=0.85,
        feature=(0.0, 0.0, 1.0),
        progress=Fraction(80, 100),
        blog_image_type="normal_gameplay",
        explanation_value="high",
        context_relevance="none",
        scene_selection_role="recurring_gameplay",
        scene_slug="armored-boss",
        combat_encounter_kind="major",
        video_order=1,
        video_fingerprint="b" * 64,
        summary="鎧を着た戦闘対象との戦闘",
        combat_subject_evidence=different_subject,
    )

    # Act
    result = select_video_set_images(
        (weaker, other_boss, best),
        requested_count=3,
        spoiler_sensitivity="medium",
        similarity_threshold=0.72,
    )
    reversed_result = select_video_set_images(
        (best, other_boss, weaker),
        requested_count=3,
        spoiler_sensitivity="medium",
        similarity_threshold=0.72,
    )

    # Assert
    assert [item.candidate.identifier for item in result.selected] == [
        best.identifier,
        other_boss.identifier,
    ]
    assert result.shortfall is True
    assert len(result.rejected) == 1
    rejection = result.rejected[0]
    assert rejection.candidate.identifier == weaker.identifier
    assert rejection.reason_code == "semantic_duplicate"
    assert rejection.blocked_by_image_id == best.identifier
    assert rejection.semantic_group_basis == "combat_subject_appearance"
    expected_group_evidence = (
        "body_plan:quadruped",
        "scale:large",
        "surface:organic",
        "color:brown",
        "color:green",
        "trait:bulbous_body",
        "trait:large_mouth",
    )
    assert rejection.semantic_group_evidence == expected_group_evidence
    representative = next(
        item for item in result.selected if item.candidate.identifier == best.identifier
    )
    assert representative.semantic_group_evidence == expected_group_evidence
    assert [item.candidate.identifier for item in reversed_result.selected] == [
        item.candidate.identifier for item in result.selected
    ]
    assert (
        reversed_result.rejected[0].reason_code,
        reversed_result.rejected[0].blocked_by_image_id,
        reversed_result.rejected[0].semantic_group_id,
        reversed_result.rejected[0].semantic_group_evidence,
    ) == (
        rejection.reason_code,
        rejection.blocked_by_image_id,
        rejection.semantic_group_id,
        rejection.semantic_group_evidence,
    )


def test_combat_subject_group_absorbs_matching_encounter_group() -> None:
    """同じ対象と同じ遭遇の根拠が一つのSemantic Groupへ統合されること。

    Arrange:
        - 同じVideo Source、遭遇、外見を持つ主要戦闘候補2件が用意される
    Act:
        - 2枚の選定が要求される
    Assert:
        - Group根拠の重複で失敗せず最高utilityの1枚だけが選択されること
        - 統合後の根拠がCombat Subject Groupになること
    """
    # Arrange
    subject = CombatSubjectEvidence(
        body_plan="serpentine",
        scale="enormous",
        surface="scaled",
        colors=("blue", "purple"),
        traits=("elongated_body", "horns"),
        distinctiveness="distinctive",
    )
    source_fingerprint = "c" * 64
    best = _candidate(
        "matching-subject-encounter-best",
        quality=0.95,
        feature=(1.0, 0.0),
        progress=Fraction(10, 100),
        blog_image_type="normal_gameplay",
        explanation_value="high",
        context_relevance="none",
        scene_selection_role="recurring_gameplay",
        scene_slug="serpent-battle",
        combat_encounter_kind="major",
        video_fingerprint=source_fingerprint,
        combat_subject_evidence=subject,
    )
    weaker = _candidate(
        "matching-subject-encounter-weaker",
        quality=0.80,
        feature=(0.90, math.sqrt(1 - 0.90**2)),
        progress=Fraction(12, 100),
        blog_image_type="normal_gameplay",
        explanation_value="high",
        context_relevance="none",
        scene_selection_role="recurring_gameplay",
        scene_slug="serpent-battle",
        combat_encounter_kind="major",
        video_fingerprint=source_fingerprint,
        combat_subject_evidence=subject,
    )

    # Act
    result = select_video_set_images(
        (weaker, best),
        requested_count=2,
        spoiler_sensitivity="medium",
        similarity_threshold=0.72,
    )

    # Assert
    assert [item.candidate.identifier for item in result.selected] == [best.identifier]
    assert result.rejected[0].candidate.identifier == weaker.identifier
    assert result.rejected[0].semantic_group_basis == "combat_subject_appearance"


def test_distinct_combat_subjects_in_one_encounter_are_not_merged() -> None:
    """同じ遭遇内でも外見が明確に異なる戦闘対象が別々に選択されること。

    Arrange:
        - 同じVideo SourceとScene Slugに外見が異なる主要戦闘対象2件がある
    Act:
        - 2枚の選定が要求される
    Assert:
        - Encounter Groupだけを理由に同じGroupへ統合されないこと
        - 両方の戦闘対象が選択されること
    """
    # Arrange
    source_fingerprint = "d" * 64
    frog = _candidate(
        "distinct-subject-frog",
        quality=0.90,
        feature=(1.0, 0.0),
        progress=Fraction(10, 100),
        blog_image_type="normal_gameplay",
        explanation_value="high",
        context_relevance="none",
        scene_selection_role="recurring_gameplay",
        scene_slug="major-battle",
        combat_encounter_kind="major",
        video_fingerprint=source_fingerprint,
        combat_subject_evidence=CombatSubjectEvidence(
            body_plan="quadruped",
            scale="large",
            surface="organic",
            colors=("green",),
            traits=("large_mouth",),
            distinctiveness="distinctive",
        ),
    )
    armored = _candidate(
        "distinct-subject-armored",
        quality=0.85,
        feature=(0.0, 1.0),
        progress=Fraction(12, 100),
        blog_image_type="normal_gameplay",
        explanation_value="high",
        context_relevance="none",
        scene_selection_role="recurring_gameplay",
        scene_slug="major-battle",
        combat_encounter_kind="major",
        video_fingerprint=source_fingerprint,
        combat_subject_evidence=CombatSubjectEvidence(
            body_plan="humanoid",
            scale="large",
            surface="armored",
            colors=("black", "red"),
            traits=("armor", "weapon"),
            distinctiveness="distinctive",
        ),
    )

    # Act
    result = select_video_set_images(
        (armored, frog),
        requested_count=2,
        spoiler_sensitivity="medium",
        similarity_threshold=0.72,
    )

    # Assert
    assert {item.candidate.identifier for item in result.selected} == {
        frog.identifier,
        armored.identifier,
    }
    assert result.rejected == ()


def test_combat_subject_matching_tolerates_independent_evidence_variation() -> None:
    """独立画像評価の一部列挙差を越えて同じ戦闘対象がまとめられること。

    Arrange:
        - 身体構造などの中核特徴が一致し色と特徴の一部だけが異なる2件がある
        - 2件のNeutral視覚特徴が同じ対象を支持する
    Act:
        - 異なるVideo Sourceから2枚の選定が要求される
    Assert:
        - 固有名や完全一致に依存せず最高utilityの1枚だけが選択されること
    """
    # Arrange
    first = _candidate(
        "varied-subject-first",
        quality=0.90,
        feature=(1.0, 0.0),
        progress=Fraction(10, 100),
        blog_image_type="normal_gameplay",
        explanation_value="high",
        context_relevance="none",
        scene_slug="incorrect-name",
        combat_encounter_kind="major",
        video_order=0,
        video_fingerprint="e" * 64,
        combat_subject_evidence=CombatSubjectEvidence(
            body_plan="quadruped",
            scale="large",
            surface="organic",
            colors=("brown", "green"),
            traits=("bulbous_body", "large_mouth"),
            distinctiveness="distinctive",
        ),
    )
    second = _candidate(
        "varied-subject-second",
        quality=0.80,
        feature=(0.90, math.sqrt(1 - 0.90**2)),
        progress=Fraction(70, 100),
        blog_image_type="other",
        explanation_value="high",
        context_relevance="none",
        scene_slug="generic-boss",
        combat_encounter_kind="major",
        video_order=1,
        video_fingerprint="f" * 64,
        combat_subject_evidence=CombatSubjectEvidence(
            body_plan="quadruped",
            scale="large",
            surface="organic",
            colors=("green",),
            traits=("large_mouth",),
            distinctiveness="distinctive",
        ),
    )

    # Act
    result = select_video_set_images(
        (second, first),
        requested_count=2,
        spoiler_sensitivity="medium",
        similarity_threshold=0.72,
    )

    # Assert
    assert [item.candidate.identifier for item in result.selected] == [first.identifier]
    assert result.rejected[0].candidate.identifier == second.identifier
    assert result.rejected[0].semantic_group_basis == "combat_subject_appearance"


def test_generic_combat_subject_evidence_does_not_merge_across_videos() -> None:
    """genericな大分類だけでは異なる動画の主要戦闘が統合されないこと。

    Arrange:
        - 同じ一般的な外見観測と似た画面を持つ異なるVideo Sourceが用意される
        - 外見観測は対象同一性を裏付けられないgenericとされる
    Act:
        - 2枚の選定が要求される
    Assert:
        - 「ボス戦」という大分類だけで同じSubject Groupにされないこと
    """
    # Arrange
    generic = CombatSubjectEvidence(
        body_plan="humanoid",
        scale="large",
        surface="armored",
        colors=("black",),
        traits=("armor",),
        distinctiveness="generic",
    )
    first = _candidate(
        "generic-subject-first",
        quality=0.90,
        feature=(1.0, 0.0),
        progress=Fraction(10, 100),
        blog_image_type="normal_gameplay",
        explanation_value="high",
        context_relevance="none",
        combat_encounter_kind="major",
        video_order=0,
        video_fingerprint="1" * 64,
        combat_subject_evidence=generic,
    )
    second = _candidate(
        "generic-subject-second",
        quality=0.85,
        feature=(0.90, math.sqrt(1 - 0.90**2)),
        progress=Fraction(70, 100),
        blog_image_type="normal_gameplay",
        explanation_value="high",
        context_relevance="none",
        combat_encounter_kind="major",
        video_order=1,
        video_fingerprint="2" * 64,
        combat_subject_evidence=generic,
    )

    # Act
    result = select_video_set_images(
        (second, first),
        requested_count=2,
        spoiler_sensitivity="medium",
        similarity_threshold=0.95,
    )

    # Assert
    assert {item.candidate.identifier for item in result.selected} == {
        first.identifier,
        second.identifier,
    }
    assert result.rejected == ()


def test_non_major_candidate_splits_repeated_major_combat_scene_slug() -> None:
    """非主要場面を挟む同名Scene Slugが別の主要戦闘として維持されること。

    Arrange:
        - 同じVideo SourceとScene Slugを持つ主要戦闘候補2件がある
        - 2件の間に非主要の探索候補がある
    Act:
        - 3枚の選定が要求される
    Assert:
        - 非主要候補が遭遇境界となり3件すべてが選択されること
        - 主要戦闘2件が同じSemantic Duplicate Groupへ割り当てられないこと
    """
    # Arrange
    source_fingerprint = "4" * 64
    first_encounter = _candidate(
        "first-repeated-major-encounter",
        quality=0.90,
        feature=(1.0, 0.0, 0.0),
        progress=Fraction(10, 100),
        blog_image_type="normal_gameplay",
        explanation_value="high",
        context_relevance="none",
        scene_slug="repeated-boss",
        combat_encounter_kind="major",
        video_fingerprint=source_fingerprint,
    )
    exploration = _candidate(
        "between-major-encounters",
        quality=0.80,
        feature=(0.0, 1.0, 0.0),
        progress=Fraction(30, 100),
        blog_image_type="normal_gameplay",
        explanation_value="high",
        context_relevance="none",
        scene_slug="exploration",
        video_fingerprint=source_fingerprint,
    )
    second_encounter = _candidate(
        "second-repeated-major-encounter",
        quality=0.85,
        feature=(0.0, 0.0, 1.0),
        progress=Fraction(60, 100),
        blog_image_type="normal_gameplay",
        explanation_value="high",
        context_relevance="none",
        scene_slug="repeated-boss",
        combat_encounter_kind="major",
        video_fingerprint=source_fingerprint,
    )

    # Act
    result = select_video_set_images(
        (second_encounter, exploration, first_encounter),
        requested_count=3,
        spoiler_sensitivity="medium",
        similarity_threshold=0.72,
    )

    # Assert
    assert {item.candidate.identifier for item in result.selected} == {
        first_encounter.identifier,
        exploration.identifier,
        second_encounter.identifier,
    }
    encounter_groups = {
        item.semantic_group_id
        for item in result.selected
        if item.candidate.annotation.combat_encounter_kind == "major"
    }
    assert encounter_groups == {None}


def test_title_semantics_limit_applies_across_blog_image_type_misclassification() -> (
    None
):
    """タイトル画面がeventへ誤分類されても最良の1枚だけが選択されること。

    Arrange:
        - 画像内根拠はタイトル画面だがBlog Image Typeがeventの高品質候補がある
        - 正しくtitleへ分類された候補と通常プレイ候補がある
    Act:
        - 3枚の選定が要求される
    Assert:
        - 画像内根拠で識別された最良のタイトル画面と通常プレイだけが選択されること
        - もう一つのタイトル画面へstable reasonとblocking selected IDが記録されること
    """
    # Arrange
    misclassified_title = _candidate(
        "misclassified-title",
        quality=0.95,
        feature=(1.0, 0.0, 0.0),
        progress=Fraction(10, 100),
        blog_image_type="event",
        explanation_value="high",
        context_relevance="none",
        scene_slug="opening-event",
        screen_text_kind="title",
        content_kind="title",
    )
    classified_title = _candidate(
        "classified-title",
        quality=0.90,
        feature=(0.0, 1.0, 0.0),
        progress=Fraction(12, 100),
        blog_image_type="title",
        explanation_value="high",
        context_relevance="none",
        scene_slug="title-screen",
        screen_text_kind="title",
        content_kind="title",
    )
    gameplay = _candidate(
        "title-test-gameplay",
        quality=0.80,
        feature=(0.0, 0.0, 1.0),
        progress=Fraction(70, 100),
        blog_image_type="normal_gameplay",
        explanation_value="high",
        context_relevance="none",
    )

    # Act
    result = select_video_set_images(
        (classified_title, gameplay, misclassified_title),
        requested_count=3,
        spoiler_sensitivity="medium",
        similarity_threshold=0.72,
    )

    # Assert
    assert {item.candidate.identifier for item in result.selected} == {
        misclassified_title.identifier,
        gameplay.identifier,
    }
    rejection = result.rejected[0]
    assert rejection.candidate.identifier == classified_title.identifier
    assert rejection.reason_code == "semantic_duplicate"
    assert rejection.blocked_by_image_id == misclassified_title.identifier
    assert rejection.semantic_group_basis == "title_semantics"


def test_semantic_movement_duplicate_does_not_displace_other_useful_roles() -> None:
    """同じ移動画面の2枚目より通常戦闘と有用eventが優先されること。

    Arrange:
        - 異なるScene SlugとVariant Groupに分かれた近接する移動画面2件がある
        - 視覚類似度だけでは許可される通常戦闘とeventがある
    Act:
        - global similarityを下げず4枚の選定が要求される
    Assert:
        - 最良の移動画面、通常戦闘、eventだけが選択されること
        - 移動画面の2枚目がsemantic duplicateとして除外されること
    """
    # Arrange
    source_fingerprint = "2" * 64
    movement_best = _candidate(
        "movement-best",
        quality=0.95,
        feature=(1.0, 0.0, 0.0, 0.0),
        progress=Fraction(10, 100),
        blog_image_type="normal_gameplay",
        explanation_value="high",
        context_relevance="none",
        scene_selection_role="recurring_gameplay",
        scene_slug="field-movement",
        video_fingerprint=source_fingerprint,
        content_kind="gameplay_action",
        summary="ダンジョン内を移動している場面",
    )
    movement_duplicate = _candidate(
        "movement-duplicate",
        quality=0.90,
        feature=(0.94, math.sqrt(1 - 0.94**2), 0.0, 0.0),
        progress=Fraction(12, 100),
        blog_image_type="other",
        explanation_value="high",
        context_relevance="none",
        scene_selection_role="recurring_gameplay",
        scene_slug="corridor-navigation",
        video_fingerprint=source_fingerprint,
        content_kind="gameplay_action",
        summary="ダンジョン内を移動している場面",
    )
    ordinary_combat = _candidate(
        "movement-test-ordinary-combat",
        quality=0.80,
        feature=(0.0, 0.0, 1.0, 0.0),
        progress=Fraction(50, 100),
        blog_image_type="normal_gameplay",
        explanation_value="high",
        context_relevance="none",
        combat_encounter_kind="ordinary",
        video_fingerprint=source_fingerprint,
        content_kind="gameplay_action",
    )
    event = _candidate(
        "movement-test-event",
        quality=0.75,
        feature=(0.0, 0.0, 0.0, 1.0),
        progress=Fraction(80, 100),
        blog_image_type="event",
        explanation_value="high",
        context_relevance="none",
        video_fingerprint=source_fingerprint,
        content_kind="event_dialogue",
    )

    # Act
    result = select_video_set_images(
        (movement_duplicate, event, movement_best, ordinary_combat),
        requested_count=4,
        spoiler_sensitivity="medium",
        similarity_threshold=0.95,
    )

    # Assert
    assert {item.candidate.identifier for item in result.selected} == {
        movement_best.identifier,
        ordinary_combat.identifier,
        event.identifier,
    }
    rejection = result.rejected[0]
    assert rejection.candidate.identifier == movement_duplicate.identifier
    assert rejection.reason_code == "semantic_duplicate"
    assert rejection.blocked_by_image_id == movement_best.identifier
    assert rejection.semantic_group_basis == "visual_role_similarity"


@pytest.mark.parametrize(
    ("first_summary", "second_summary"),
    (
        ("炎の技で敵を攻撃している場面", "氷の技で敵を攻撃している場面"),
        ("［…］", "［…］"),
    ),
)
def test_useful_recurring_gameplay_variants_remain_selectable(
    first_summary: str,
    second_summary: str,
) -> None:
    """別説明または情報のないsummaryだけで有用variantが失われないこと。

    Arrange:
        - 同じScene Slug、画像内役割、近接時刻、高い視覚類似度を持つ2候補がある
        - 各候補の独立した画像説明は異なる技または情報のないredactionを示す
    Act:
        - recurring gameplay候補2枚の選定が要求される
    Assert:
        - 追加説明価値を持つ両方のvariantが選択されること
    """
    # Arrange
    source_fingerprint = "3" * 64
    fire_action = _candidate(
        "fire-action-variant",
        quality=0.90,
        feature=(1.0, 0.0),
        progress=Fraction(20, 100),
        blog_image_type="normal_gameplay",
        explanation_value="high",
        context_relevance="none",
        scene_selection_role="recurring_gameplay",
        scene_slug="ordinary-battle",
        video_fingerprint=source_fingerprint,
        content_kind="gameplay_action",
        summary=first_summary,
    )
    ice_action = _candidate(
        "ice-action-variant",
        quality=0.85,
        feature=(0.94, math.sqrt(1 - 0.94**2)),
        progress=Fraction(22, 100),
        blog_image_type="normal_gameplay",
        explanation_value="high",
        context_relevance="none",
        scene_selection_role="recurring_gameplay",
        scene_slug="ordinary-battle",
        video_fingerprint=source_fingerprint,
        content_kind="gameplay_action",
        summary=second_summary,
    )

    # Act
    result = select_video_set_images(
        (ice_action, fire_action),
        requested_count=2,
        spoiler_sensitivity="medium",
        similarity_threshold=0.95,
    )

    # Assert
    assert {item.candidate.identifier for item in result.selected} == {
        fire_action.identifier,
        ice_action.identifier,
    }
    assert result.rejected == ()
