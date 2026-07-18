"""Candidate Frame Observationの決定的な正規化を検証する。"""

import pytest

from src.video_selection.models.candidate_annotation import (
    BlogImageType,
    ExplanationValue,
)
from src.video_selection.models.candidate_frame_observation import (
    CandidateFrameContentKind,
    CandidateFrameObservation,
    CandidateInterfaceKind,
)
from src.video_selection.models.frame_candidate import FrameCandidate


@pytest.mark.parametrize(
    (
        "content_kind",
        "interface_kind",
        "visible_dialogue_text",
        "visible_action",
        "visible_character_or_enemy",
        "expected_content_kind",
        "expected_explanation_value",
        "expected_blog_image_type",
    ),
    (
        (
            "event_dialogue",
            "tutorial_help",
            True,
            False,
            False,
            "tutorial_help",
            "none",
            "menu",
        ),
        (
            "other_interface",
            "save",
            False,
            False,
            False,
            "save",
            "none",
            "menu",
        ),
        (
            "shop",
            "shop",
            False,
            False,
            False,
            "shop",
            "none",
            "menu",
        ),
        (
            "event_dialogue",
            "none",
            False,
            False,
            True,
            "event_setup",
            "none",
            "event",
        ),
        (
            "event_action",
            "other_interface",
            False,
            True,
            True,
            "event_action",
            "high",
            "event",
        ),
    ),
)
def test_atomic_observations_normalize_ambiguous_model_content(
    content_kind: CandidateFrameContentKind,
    interface_kind: CandidateInterfaceKind,
    visible_dialogue_text: bool,
    visible_action: bool,
    visible_character_or_enemy: bool,
    expected_content_kind: CandidateFrameContentKind,
    expected_explanation_value: ExplanationValue,
    expected_blog_image_type: BlogImageType,
) -> None:
    """単純な視覚観測から曖昧なmodel分類が決定的に正規化されること。

    Arrange:
        - 高評価だがinterface・台詞・動作の関係が異なるframe観測が用意される
    Act:
        - 観測の決定的な公開値が参照される
    Assert:
        - 動作中の戦闘を保ち、静止UIと台詞のないeventが掲載不可にされること
    """
    # Arrange
    observation = CandidateFrameObservation(
        candidate=FrameCandidate("frm_" + "a" * 64, b"image"),
        scene_slug="scene",
        content_kind=content_kind,
        interface_kind=interface_kind,
        visible_dialogue_text=visible_dialogue_text,
        visible_action=visible_action,
        visible_character_or_enemy=visible_character_or_enemy,
        explanation_value="high",
        screen_text_kind="dialogue",
        primary_subject_visibility="clear",
        transient_obstruction="none",
        spoiler_risk="none",
        spoiler_evidence="",
    )

    # Act
    effective_content_kind = observation.effective_content_kind
    explanation_value = observation.effective_explanation_value
    blog_image_type = observation.blog_image_type

    # Assert
    assert effective_content_kind == expected_content_kind
    assert explanation_value == expected_explanation_value
    assert blog_image_type == expected_blog_image_type
