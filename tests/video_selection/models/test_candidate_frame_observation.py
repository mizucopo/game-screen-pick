"""Candidate Frame Observationの決定的な正規化を検証する。"""

from src.video_selection.models.candidate_frame_observation import (
    CandidateFrameObservation,
)
from src.video_selection.models.frame_candidate import FrameCandidate


def test_tutorial_help_is_normalized_to_no_explanation_value() -> None:
    """modelの過大評価にかかわらずtutorialが説明価値なしにされること。

    Arrange:
        - mediumと評価されたtutorial_helpのframe別観測が用意される
    Act:
        - 観測の決定的な公開値が参照される
    Assert:
        - menu画像として扱われ、説明価値なしとmenu textへ正規化されること
    """
    # Arrange
    observation = CandidateFrameObservation(
        candidate=FrameCandidate("frm_" + "a" * 64, b"image"),
        scene_slug="scene",
        content_kind="tutorial_help",
        explanation_value="medium",
        screen_text_kind="dialogue",
        primary_subject_visibility="clear",
        transient_obstruction="none",
        spoiler_risk="none",
        spoiler_evidence="",
    )

    # Act
    explanation_value = observation.effective_explanation_value
    blog_image_type = observation.blog_image_type
    screen_text_kind = observation.effective_screen_text_kind

    # Assert
    assert explanation_value == "none"
    assert blog_image_type == "menu"
    assert screen_text_kind == "menu"
