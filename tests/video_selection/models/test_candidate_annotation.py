import pytest

from src.video_selection.models.candidate_annotation import CandidateAnnotation
from src.video_selection.models.frame_candidate import FrameCandidate


def test_major_spoiler_requires_safe_evidence_summary() -> None:
    """high Spoiler Riskに引用ではないevidence summaryが要求されること。

    Arrange:
        - high riskと短い意味証拠を持つCandidate Annotationが用意される
    Act:
        - Candidate Annotationが構築される
    Assert:
        - riskとevidenceが分離して保持されること
    """
    # Arrange
    candidate = FrameCandidate(identifier="frame-1", image_bytes=b"image")

    # Act
    annotation = CandidateAnnotation(
        candidate=candidate,
        summary="終盤の対決場面",
        candidate_moment_id="mom_" + "a" * 64,
        scene_slug="climax",
        blog_image_type="event",
        explanation_value="high",
        frame_choice_reason="対決する人物が明確に写る",
        screen_text_kind="dialogue",
        context_relevance="strong",
        supporting_context_cue_ids=("cue-1",),
        spoiler_risk="high",
        spoiler_evidence="主要人物の正体が明示される",
    )

    # Assert
    assert annotation.candidate is candidate
    assert annotation.spoiler_risk == "high"
    assert annotation.spoiler_evidence == "主要人物の正体が明示される"
    assert not hasattr(annotation, "quality_score")
    assert not hasattr(annotation, "final_score")
    assert not hasattr(annotation, "selected")


def test_ordinary_combat_and_event_expose_distinct_selection_coverage_facets() -> None:
    """通常戦闘とイベントから条件付きcoverage facetが導出されること。

    Arrange:
        - Spoiler Riskを独立に持つ通常戦闘、主要戦闘、判別不能戦闘が用意される
        - イベントのannotationが用意される
    Act:
        - 各annotationのSelection Coverage Facetが読み出される
    Assert:
        - 通常戦闘とイベントだけが対応facetを返すこと
    """
    # Arrange
    candidate = FrameCandidate(identifier="frame-1", image_bytes=b"image")
    ordinary_combat = CandidateAnnotation(
        candidate=candidate,
        summary="通常戦闘",
        blog_image_type="normal_gameplay",
        explanation_value="high",
        combat_encounter_kind="ordinary",
        spoiler_risk="medium",
        spoiler_evidence="物語上の進行情報が表示される",
    )
    major_combat = CandidateAnnotation(
        candidate=candidate,
        summary="主要戦闘",
        blog_image_type="normal_gameplay",
        explanation_value="high",
        combat_encounter_kind="major",
    )
    uncertain_combat = CandidateAnnotation(
        candidate=candidate,
        summary="判別不能な戦闘",
        blog_image_type="normal_gameplay",
        explanation_value="high",
        combat_encounter_kind="uncertain",
    )
    event = CandidateAnnotation(
        candidate=candidate,
        summary="イベント",
        blog_image_type="event",
        explanation_value="high",
    )

    # Act
    ordinary_facet = ordinary_combat.selection_coverage_facet
    major_facet = major_combat.selection_coverage_facet
    uncertain_facet = uncertain_combat.selection_coverage_facet
    event_facet = event.selection_coverage_facet

    # Assert
    assert ordinary_facet == "ordinary_combat"
    assert major_facet is None
    assert uncertain_facet is None
    assert event_facet == "event"
    assert ordinary_combat.combat_action is True
    assert major_combat.combat_action is True
    assert uncertain_combat.combat_action is True
    assert event.combat_action is False


def test_combat_encounter_kind_requires_a_known_value() -> None:
    """未知のCombat Encounter Kindならannotationが拒否されること。

    Arrange:
        - 未知のcombat_encounter_kindを持つannotation入力が用意される
    Act:
        - Candidate Annotationの構築が試行される
    Assert:
        - domain field不正として拒否されること
    """
    # Arrange
    candidate = FrameCandidate(identifier="frame-1", image_bytes=b"image")

    # Act
    # Assert
    with pytest.raises(ValueError, match="domain field"):
        CandidateAnnotation(
            candidate=candidate,
            summary="通常戦闘",
            combat_encounter_kind="boss",  # type: ignore[arg-type]
        )
