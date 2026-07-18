"""frame別観測からRepresentative Frameを決定するserviceのtest。"""

from fractions import Fraction

from src.video_selection.models.candidate_annotation import (
    ExplanationValue,
    ScreenTextKind,
)
from src.video_selection.models.candidate_frame_observation import (
    CandidateFrameContentKind,
    CandidateFrameObservation,
    CandidateInterfaceKind,
)
from src.video_selection.models.frame_candidate import FrameCandidate
from src.video_selection.models.neutral_image_analysis import NeutralImageAnalysis
from src.video_selection.models.neutral_image_metrics import NeutralImageMetrics
from src.video_selection.services import (
    select_representative_candidate_frame_observation as selection_service,
)


def test_meaningful_event_is_selected_over_equal_shop_and_idle_frames() -> None:
    """同じ説明価値ならinterfaceや待機画面より出来事の見えるframeが選ばれること。

    Arrange:
        - shop、待機中gameplay、台詞eventの3つのframe別観測が用意される
    Act:
        - Representative Frameが決定的に選択される
    Assert:
        - neutral画質が少し低くても台詞eventのframeが選ばれること
    """
    # Arrange
    observations = (
        _observation("a", "shop", quality=0.90, explanation_value="high"),
        _observation(
            "b",
            "gameplay_idle",
            quality=0.95,
            explanation_value="high",
        ),
        _observation(
            "c",
            "event_dialogue",
            quality=0.80,
            explanation_value="high",
            screen_text_kind="dialogue",
        ),
    )

    # Act
    selected = selection_service.select_representative_candidate_frame_observation(
        observations
    )

    # Assert
    assert selected.candidate.identifier == "frm_" + "c" * 64


def test_grossly_degraded_frame_cannot_override_visible_peer() -> None:
    """意味評価が高くても著しく見づらいframeが代表にされないこと。

    Arrange:
        - 文脈上はhighだが低画質なoverlayと、明瞭なgameplay frameが用意される
    Act:
        - Representative Frameが決定的に選択される
    Assert:
        - 画質・可視性・情報量が同時に大幅低下したoverlayが除外されること
    """
    # Arrange
    observations = (
        _observation(
            "a",
            "event_action",
            quality=0.20,
            information=0.05,
            visibility=0.75,
            explanation_value="high",
        ),
        _observation(
            "b",
            "gameplay_action",
            quality=0.82,
            information=0.70,
            visibility=0.96,
            explanation_value="medium",
        ),
    )

    # Act
    selected = selection_service.select_representative_candidate_frame_observation(
        observations
    )

    # Assert
    assert selected.candidate.identifier == "frm_" + "b" * 64


def test_static_tutorial_cannot_override_action_frame() -> None:
    """modelがeventと誤分類した説明画面より動作frameが選ばれること。

    Arrange:
        - highのevent_dialogueとされた静止tutorialとmediumの戦闘が用意される
    Act:
        - Representative Frameが決定的に選択される
    Assert:
        - atomic observationで掲載不可になるtutorialが選ばれないこと
    """
    # Arrange
    tutorial = _observation(
        "a",
        "event_dialogue",
        quality=0.90,
        explanation_value="high",
        screen_text_kind="dialogue",
        interface_kind="tutorial_help",
        visible_dialogue_text=True,
        visible_character_or_enemy=False,
    )
    action = _observation(
        "b",
        "event_action",
        quality=0.80,
        explanation_value="medium",
        interface_kind="other_interface",
        visible_action=True,
    )

    # Act
    selected = selection_service.select_representative_candidate_frame_observation(
        (tutorial, action)
    )

    # Assert
    assert selected.candidate.identifier == "frm_" + "b" * 64


def _observation(
    digest_character: str,
    content_kind: CandidateFrameContentKind,
    *,
    quality: float,
    information: float = 0.70,
    visibility: float = 0.95,
    explanation_value: ExplanationValue,
    screen_text_kind: ScreenTextKind = "none",
    interface_kind: CandidateInterfaceKind = "none",
    visible_dialogue_text: bool | None = None,
    visible_action: bool | None = None,
    visible_character_or_enemy: bool = True,
) -> CandidateFrameObservation:
    metrics = NeutralImageMetrics(
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
        information_score=information,
        visibility_score=visibility,
    )
    candidate = FrameCandidate(
        identifier="frm_" + digest_character * 64,
        image_bytes=digest_character.encode(),
        video_fingerprint=digest_character * 64,
        stream_index=0,
        source_pts=1,
        origin_pts=0,
        time_base=Fraction(1, 1000),
        video_time=Fraction(1),
        analysis=NeutralImageAnalysis(
            source_pts=1,
            metrics=metrics,
            quality_score=quality,
            visual_feature=(1.0, 0.0),
            grayscale_signature=b"signature",
            reject_reason=None,
        ),
    )
    return CandidateFrameObservation(
        candidate=candidate,
        scene_slug="scene",
        content_kind=content_kind,
        interface_kind=interface_kind,
        visible_dialogue_text=(
            content_kind == "event_dialogue"
            if visible_dialogue_text is None
            else visible_dialogue_text
        ),
        visible_action=(
            content_kind in {"gameplay_action", "event_action"}
            if visible_action is None
            else visible_action
        ),
        visible_character_or_enemy=visible_character_or_enemy,
        explanation_value=explanation_value,
        screen_text_kind=screen_text_kind,
        primary_subject_visibility="clear",
        transient_obstruction="none",
        spoiler_risk="none",
        spoiler_evidence="",
    )
