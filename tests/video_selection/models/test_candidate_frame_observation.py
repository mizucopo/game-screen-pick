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
    CharacterBodyVisibility,
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
        prominent_event_portrait=False,
        cinematic_event_presentation=False,
        visible_dialogue_text=visible_dialogue_text,
        dialogue_text_presentation=(
            "dialogue_box" if visible_dialogue_text else "none"
        ),
        visible_action=visible_action,
        visible_character_or_enemy=visible_character_or_enemy,
        combat_action=False,
        player_body_visibility=("clear" if visible_character_or_enemy else "absent"),
        opponent_body_visibility="absent",
        effect_only_frame=False,
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


def test_combat_without_visible_opponent_has_no_explanation_value() -> None:
    """敵本体を判別できない戦闘がブログ掲載価値なしにされること。

    Arrange:
        - 戦闘中だが発光で敵本体が判別できない高評価frameが用意される
    Act:
        - 観測の決定的なExplanation Valueが参照される
    Assert:
        - playerだけ見える戦闘が説明価値なしにされること
    """
    # Arrange
    observation = CandidateFrameObservation(
        candidate=FrameCandidate("frm_" + "a" * 64, b"image"),
        scene_slug="battle",
        content_kind="event_action",
        interface_kind="other_interface",
        prominent_event_portrait=False,
        cinematic_event_presentation=False,
        visible_dialogue_text=False,
        dialogue_text_presentation="none",
        visible_action=True,
        visible_character_or_enemy=True,
        combat_action=True,
        player_body_visibility="clear",
        opponent_body_visibility="absent",
        effect_only_frame=False,
        explanation_value="high",
        screen_text_kind="hud",
        primary_subject_visibility="clear",
        transient_obstruction="none",
        spoiler_risk="none",
        spoiler_evidence="",
    )

    # Act
    explanation_value = observation.effective_explanation_value

    # Assert
    assert explanation_value == "none"


@pytest.mark.parametrize(
    ("opponent_body_visibility", "effect_only_frame"),
    (("partial", False), ("absent", False), ("clear", True)),
)
def test_unreadable_action_frame_has_no_explanation_value(
    opponent_body_visibility: CharacterBodyVisibility,
    effect_only_frame: bool,
) -> None:
    """敵本体が不明瞭またはエフェクトだけの動作frameが掲載不可にされること。

    Arrange:
        - 敵本体が不明瞭か、一時的なエフェクトだけが主内容の高評価frameが用意される
    Act:
        - 観測の決定的なExplanation Valueが参照される
    Assert:
        - modelの高評価より直接観測が優先され、説明価値なしにされること
    """
    # Arrange
    observation = CandidateFrameObservation(
        candidate=FrameCandidate("frm_" + "b" * 64, b"image"),
        scene_slug="battle",
        content_kind="event_action",
        interface_kind="none",
        prominent_event_portrait=False,
        cinematic_event_presentation=False,
        visible_dialogue_text=False,
        dialogue_text_presentation="none",
        visible_action=True,
        visible_character_or_enemy=True,
        combat_action=True,
        player_body_visibility="partial",
        opponent_body_visibility=opponent_body_visibility,
        effect_only_frame=effect_only_frame,
        explanation_value="high",
        screen_text_kind="hud",
        primary_subject_visibility="clear",
        transient_obstruction="none",
        spoiler_risk="none",
        spoiler_evidence="",
    )

    # Act
    explanation_value = observation.effective_explanation_value

    # Assert
    assert explanation_value == "none"


@pytest.mark.parametrize(
    (
        "content_kind",
        "interface_kind",
        "prominent_event_portrait",
        "cinematic_event_presentation",
        "expected_content_kind",
        "expected_blog_image_type",
    ),
    (
        ("other", "document", False, False, "document", "menu"),
        ("gameplay_idle", "none", True, False, "event_setup", "event"),
        ("gameplay_idle", "none", False, True, "event_setup", "event"),
    ),
)
def test_static_document_and_silent_event_presentation_have_no_explanation_value(
    content_kind: CandidateFrameContentKind,
    interface_kind: CandidateInterfaceKind,
    prominent_event_portrait: bool,
    cinematic_event_presentation: bool,
    expected_content_kind: CandidateFrameContentKind,
    expected_blog_image_type: BlogImageType,
) -> None:
    """静止文書と台詞のないイベント演出が掲載不可にされること。

    Arrange:
        - 高評価だが静止文書または台詞のないイベント演出の観測が用意される
    Act:
        - 観測の決定的な公開値が参照される
    Assert:
        - 画面種別が補正され、説明価値なしにされること
    """
    # Arrange
    observation = CandidateFrameObservation(
        candidate=FrameCandidate("frm_" + "a" * 64, b"image"),
        scene_slug="scene",
        content_kind=content_kind,
        interface_kind=interface_kind,
        prominent_event_portrait=prominent_event_portrait,
        cinematic_event_presentation=cinematic_event_presentation,
        visible_dialogue_text=False,
        dialogue_text_presentation="none",
        visible_action=False,
        visible_character_or_enemy=True,
        combat_action=False,
        player_body_visibility="clear",
        opponent_body_visibility="absent",
        effect_only_frame=False,
        explanation_value="high",
        screen_text_kind="none",
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
    assert explanation_value == "none"
    assert blog_image_type == expected_blog_image_type


def test_visible_event_dialogue_overrides_generic_interface() -> None:
    """人物立ち絵と台詞を持つ会話eventが汎用UIより優先されること。

    Arrange:
        - 大きな人物立ち絵と台詞が見える一方で汎用UIと分類された観測が用意される
    Act:
        - 観測の決定的な公開値が参照される
    Assert:
        - 会話event、台詞、元の説明価値へ正規化されること
    """
    # Arrange
    observation = CandidateFrameObservation(
        candidate=FrameCandidate("frm_" + "a" * 64, b"image"),
        scene_slug="event",
        content_kind="other_interface",
        interface_kind="other_interface",
        prominent_event_portrait=True,
        cinematic_event_presentation=False,
        visible_dialogue_text=True,
        dialogue_text_presentation="dialogue_box",
        visible_action=False,
        visible_character_or_enemy=True,
        combat_action=False,
        player_body_visibility="clear",
        opponent_body_visibility="absent",
        effect_only_frame=False,
        explanation_value="high",
        screen_text_kind="menu",
        primary_subject_visibility="clear",
        transient_obstruction="none",
        spoiler_risk="none",
        spoiler_evidence="",
    )

    # Act
    effective_content_kind = observation.effective_content_kind
    blog_image_type = observation.blog_image_type
    screen_text_kind = observation.effective_screen_text_kind
    explanation_value = observation.effective_explanation_value

    # Assert
    assert effective_content_kind == "event_dialogue"
    assert blog_image_type == "event"
    assert screen_text_kind == "dialogue"
    assert explanation_value == "high"


def test_dialogue_visibility_requires_a_visible_text_presentation() -> None:
    """画面内台詞の真偽値と視覚的な表示形式が一致させられること。

    Arrange:
        - 台詞が見えるとする一方で表示形式がないframe観測が用意される
    Act:
        - Candidate Frame Observationが構築される
    Assert:
        - 矛盾した直接観測が拒否されること
    """
    # Arrange
    candidate = FrameCandidate("frm_" + "a" * 64, b"image")

    # Act
    with pytest.raises(ValueError) as raised:
        CandidateFrameObservation(
            candidate=candidate,
            scene_slug="scene",
            content_kind="event_dialogue",
            interface_kind="none",
            prominent_event_portrait=False,
            cinematic_event_presentation=True,
            visible_dialogue_text=True,
            dialogue_text_presentation="none",
            visible_action=False,
            visible_character_or_enemy=True,
            combat_action=False,
            player_body_visibility="clear",
            opponent_body_visibility="absent",
            effect_only_frame=False,
            explanation_value="high",
            screen_text_kind="dialogue",
            primary_subject_visibility="clear",
            transient_obstruction="none",
            spoiler_risk="none",
            spoiler_evidence="",
        )

    # Assert
    assert str(raised.value) == "Candidate Frame Observationのdomain fieldが不正です"
