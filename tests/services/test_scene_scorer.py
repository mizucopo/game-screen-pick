"""SceneScorerの単体テスト.

CLIP 風の固定埋め込みとレイアウトヒューリスティクスを使い、
gameplay / event / other の3分類と遷移フレーム抑制が期待通りに働くかを確認する。
"""

import pytest
import torch

from src.services.scene_scorer import SceneScorer
from tests.conftest import create_adaptive_scores, create_analyzed_image
from tests.dummy_model_manager import DummyModelManager


def _adaptive(
    *,
    distinctiveness_score: float = 0.5,
    information_score: float = 0.8,
    visibility_score: float = 0.8,
):
    return create_adaptive_scores(
        information_score=information_score,
        distinctiveness_score=distinctiveness_score,
        visibility_score=visibility_score,
    )


def test_scene_scorer_prefers_gameplay_for_gameplay_like_image() -> None:
    """通常画面ではgameplay_scoreが最も高くなること."""
    scorer = SceneScorer(DummyModelManager())
    image = create_analyzed_image(
        path="/tmp/gameplay.jpg",
        clip_features=torch.tensor([1.0, 0.0, 0.0]).numpy(),
        combined_features=torch.tensor([1.0, 0.0, 0.0, 0.0]).numpy(),
        layout_dict={"dialogue_overlay_score": 0.0, "menu_layout_score": 0.0},
    )

    assessment = scorer.assess(image, _adaptive())

    assert assessment.scene_label.value == "gameplay"
    assert assessment.gameplay_score > assessment.event_score
    assert assessment.gameplay_score > assessment.other_score
    assert assessment.transition_suppressed_event is False


def test_scene_scorer_rewards_frequent_gameplay_cluster() -> None:
    """頻出クラスタに属する画像ほどgameplayへ寄ること."""
    scorer = SceneScorer(DummyModelManager())
    image = create_analyzed_image(
        path="/tmp/gameplay_cluster.jpg",
        clip_features=torch.tensor([0.6, 0.0, 0.0]).numpy(),
        combined_features=torch.tensor([0.6, 0.0, 0.0, 0.0]).numpy(),
    )

    frequent = scorer.assess(
        image,
        _adaptive(distinctiveness_score=0.1),
    )
    rare = scorer.assess(
        image,
        _adaptive(distinctiveness_score=0.9),
    )

    assert frequent.scene_label.value == "gameplay"
    assert frequent.gameplay_score > frequent.other_score
    assert frequent.gameplay_score > rare.gameplay_score


def test_scene_scorer_prefers_event_for_boss_intro_like_image() -> None:
    """会話なしの演出シーンでもeventへ寄ること."""
    scorer = SceneScorer(DummyModelManager())
    image = create_analyzed_image(
        path="/tmp/boss_intro.jpg",
        clip_features=torch.tensor([0.0, 1.0, 0.0]).numpy(),
        combined_features=torch.tensor([0.0, 1.0, 0.0, 0.0]).numpy(),
        normalized_metrics_dict={
            "action_intensity": 0.2,
            "ui_density": 0.2,
            "dramatic_score": 0.9,
            "color_richness": 0.7,
        },
        layout_dict={"dialogue_overlay_score": 0.0, "menu_layout_score": 0.0},
    )

    assessment = scorer.assess(image, _adaptive())

    assert assessment.scene_label.value == "event"
    assert assessment.event_score > assessment.gameplay_score
    assert assessment.event_score > assessment.other_score
    assert assessment.transition_suppressed_event is False


def test_scene_scorer_keeps_event_when_event_raw_winner_is_close_to_other() -> None:
    """event が生スコア最大なら other fallback で潰れないこと."""
    scorer = SceneScorer(DummyModelManager())
    image = create_analyzed_image(
        path="/tmp/event_close_to_other.jpg",
        clip_features=torch.tensor([0.0, 0.36, 0.35]).numpy(),
        combined_features=torch.tensor([0.0, 0.36, 0.35, 0.0]).numpy(),
        normalized_metrics_dict={
            "action_intensity": 0.4,
            "ui_density": 0.35,
            "dramatic_score": 0.2,
            "color_richness": 0.2,
        },
        layout_dict={"dialogue_overlay_score": 0.2},
    )

    assessment = scorer.assess(
        image,
        _adaptive(distinctiveness_score=0.3),
    )

    assert assessment.event_score > assessment.other_score
    assert assessment.event_score - assessment.other_score <= 0.06
    assert assessment.scene_label.value == "event"
    assert assessment.transition_suppressed_event is False


def test_scene_scorer_promotes_gameplay_near_miss_to_event() -> None:
    """演出寄りの gameplay near miss だけ event へ昇格すること."""
    scorer = SceneScorer(DummyModelManager())
    image = create_analyzed_image(
        path="/tmp/gameplay_near_miss.jpg",
        raw_metrics_dict={
            "luminance_range": 60.0,
            "dominant_tone_ratio": 0.20,
        },
        clip_features=torch.tensor([0.436, 0.178, 0.14]).numpy(),
        combined_features=torch.tensor([0.436, 0.178, 0.14, 0.0]).numpy(),
        normalized_metrics_dict={
            "action_intensity": 0.35,
            "ui_density": 0.35,
            "dramatic_score": 0.8,
            "color_richness": 0.5,
        },
        layout_dict={"dialogue_overlay_score": 0.0, "menu_layout_score": 0.0},
    )

    assessment = scorer.assess(
        image,
        _adaptive(
            distinctiveness_score=0.75,
            information_score=1.0,
            visibility_score=1.0,
        ),
    )

    assert assessment.gameplay_score > assessment.event_score
    assert assessment.gameplay_score - assessment.event_score <= 0.03
    assert assessment.event_score >= 0.40
    assert assessment.other_score <= assessment.event_score - 0.01
    assert assessment.scene_label.value == "event"
    assert assessment.transition_suppressed_event is False


@pytest.mark.parametrize(
    ("distinctiveness_score", "normalized_metrics_dict"),
    [
        (
            0.55,
            {
                "action_intensity": 0.35,
                "ui_density": 0.35,
                "dramatic_score": 0.8,
                "color_richness": 0.5,
            },
        ),
        (
            0.75,
            {
                "action_intensity": 0.50,
                "ui_density": 0.35,
                "dramatic_score": 0.8,
                "color_richness": 0.5,
            },
        ),
        (
            0.75,
            {
                "action_intensity": 0.35,
                "ui_density": 0.50,
                "dramatic_score": 0.8,
                "color_richness": 0.5,
            },
        ),
    ],
)
def test_scene_scorer_does_not_promote_gameplay_near_miss_without_strong_event_signal(
    distinctiveness_score: float,
    normalized_metrics_dict: dict[str, float],
) -> None:
    """弱い signal の gameplay near miss は event に昇格しないこと."""
    scorer = SceneScorer(DummyModelManager())
    image = create_analyzed_image(
        path="/tmp/gameplay_not_promoted.jpg",
        clip_features=torch.tensor([0.436, 0.18, 0.14]).numpy(),
        combined_features=torch.tensor([0.436, 0.18, 0.14, 0.0]).numpy(),
        normalized_metrics_dict=normalized_metrics_dict,
        layout_dict={"dialogue_overlay_score": 0.0, "menu_layout_score": 0.0},
    )

    assessment = scorer.assess(
        image,
        _adaptive(distinctiveness_score=distinctiveness_score),
    )

    assert assessment.gameplay_score > assessment.event_score
    assert assessment.scene_label.value == "gameplay"


@pytest.mark.parametrize(
    ("screen_name", "layout_dict"),
    [
        ("map", {"menu_layout_score": 0.3}),
        ("equipment", {"menu_layout_score": 0.5}),
        ("shop", {"menu_layout_score": 0.4}),
        ("result_reward", {"menu_layout_score": 0.4, "title_layout_score": 0.2}),
    ],
)
def test_scene_scorer_prefers_other_for_support_ui_screens(
    screen_name: str,
    layout_dict: dict[str, float],
) -> None:
    """補助UIや遷移画面はotherへ寄ること."""
    scorer = SceneScorer(DummyModelManager())
    image = create_analyzed_image(
        path=f"/tmp/{screen_name}.jpg",
        clip_features=torch.tensor([0.0, 0.0, 1.0]).numpy(),
        combined_features=torch.tensor([0.0, 0.0, 1.0, 0.0]).numpy(),
        normalized_metrics_dict={"action_intensity": 0.1, "ui_density": 0.9},
        layout_dict=layout_dict,
    )

    assessment = scorer.assess(
        image,
        _adaptive(distinctiveness_score=0.8),
    )

    assert assessment.scene_label.value == "other"
    assert assessment.other_score > assessment.gameplay_score
    assert assessment.other_score > assessment.event_score


def test_scene_scorer_keeps_support_ui_near_miss_as_other() -> None:
    """support UI が強い other near miss は event に昇格しないこと."""
    scorer = SceneScorer(DummyModelManager())
    image = create_analyzed_image(
        path="/tmp/support_ui_near_miss.jpg",
        clip_features=torch.tensor([0.18, 0.21, 0.261]).numpy(),
        combined_features=torch.tensor([0.18, 0.21, 0.261, 0.0]).numpy(),
        normalized_metrics_dict={
            "action_intensity": 0.25,
            "ui_density": 0.75,
            "dramatic_score": 0.6,
            "color_richness": 0.5,
        },
    )

    assessment = scorer.assess(
        image,
        _adaptive(distinctiveness_score=0.7),
    )

    assert assessment.other_score > assessment.event_score
    assert assessment.scene_label.value == "other"


def test_scene_scorer_falls_back_to_other_for_ambiguous_gameplay_and_other() -> None:
    """gameplay と other が僅差なら other に倒すこと."""
    scorer = SceneScorer(DummyModelManager())
    image = create_analyzed_image(
        path="/tmp/ambiguous.jpg",
        clip_features=torch.tensor([0.73, 0.0, 0.65]).numpy(),
        combined_features=torch.tensor([0.73, 0.0, 0.65, 0.0]).numpy(),
    )

    assessment = scorer.assess(
        image,
        _adaptive(distinctiveness_score=0.5),
    )

    assert assessment.gameplay_score > assessment.other_score
    assert assessment.gameplay_score - assessment.other_score <= 0.05
    assert assessment.scene_label.value == "other"
    assert assessment.scene_confidence == 0.0


def test_scene_scorer_suppresses_low_visibility_raw_event_transition() -> None:
    """低可視性・低情報量の raw event は event から外れること."""
    scorer = SceneScorer(DummyModelManager())
    image = create_analyzed_image(
        path="/tmp/washed_transition.jpg",
        raw_metrics_dict={
            "brightness": 232.0,
            "contrast": 4.0,
            "edge_density": 0.02,
            "luminance_entropy": 0.55,
            "luminance_range": 6.0,
            "near_white_ratio": 0.68,
            "dominant_tone_ratio": 0.92,
        },
        clip_features=torch.tensor([0.10, 0.52, 0.30]).numpy(),
        combined_features=torch.tensor([0.10, 0.52, 0.30, 0.0]).numpy(),
        normalized_metrics_dict={
            "action_intensity": 0.15,
            "ui_density": 0.10,
            "dramatic_score": 0.30,
            "color_richness": 0.25,
        },
        layout_dict={"dialogue_overlay_score": 0.12, "menu_layout_score": 0.3},
    )

    assessment = scorer.assess(
        image,
        _adaptive(
            distinctiveness_score=0.62,
            information_score=0.18,
            visibility_score=0.18,
        ),
    )

    assert assessment.veiled_transition_score >= 0.34
    assert assessment.bright_washout_score >= 0.52
    assert assessment.scene_label.value != "event"
    assert assessment.transition_suppressed_event is True


def test_scene_scorer_keeps_bright_high_information_event() -> None:
    """明るい演出でも高情報量なら event のまま残ること."""
    scorer = SceneScorer(DummyModelManager())
    image = create_analyzed_image(
        path="/tmp/bright_event.jpg",
        raw_metrics_dict={
            "brightness": 218.0,
            "contrast": 18.0,
            "edge_density": 0.16,
            "luminance_entropy": 1.4,
            "luminance_range": 42.0,
            "near_white_ratio": 0.30,
            "dominant_tone_ratio": 0.58,
        },
        clip_features=torch.tensor([0.10, 0.82, 0.04]).numpy(),
        combined_features=torch.tensor([0.10, 0.82, 0.04, 0.0]).numpy(),
        normalized_metrics_dict={
            "action_intensity": 0.20,
            "ui_density": 0.20,
            "dramatic_score": 0.85,
            "color_richness": 0.80,
        },
        layout_dict={"dialogue_overlay_score": 0.55, "menu_layout_score": 0.0},
    )

    assessment = scorer.assess(
        image,
        _adaptive(
            distinctiveness_score=0.72,
            information_score=0.85,
            visibility_score=0.82,
        ),
    )

    assert assessment.scene_label.value == "event"
    assert assessment.transition_suppressed_event is False
    assert assessment.transition_risk_score < 0.72
    assert assessment.veiled_transition_score < 0.34
    assert assessment.event_score > assessment.gameplay_score


def test_scene_scorer_suppresses_bright_washed_out_dialogue_event() -> None:
    """会話UIがあっても明転中なら event から外れること."""
    scorer = SceneScorer(DummyModelManager())
    image = create_analyzed_image(
        path="/tmp/washed_out_dialogue_event.jpg",
        raw_metrics_dict={
            "brightness": 225.0,
            "contrast": 5.0,
            "edge_density": 0.03,
            "luminance_entropy": 0.65,
            "luminance_range": 8.0,
            "near_white_ratio": 0.60,
            "dominant_tone_ratio": 0.88,
        },
        clip_features=torch.tensor([0.12, 0.44, 0.31]).numpy(),
        combined_features=torch.tensor([0.12, 0.44, 0.31, 0.0]).numpy(),
        normalized_metrics_dict={
            "action_intensity": 0.18,
            "ui_density": 0.18,
            "dramatic_score": 0.40,
            "color_richness": 0.45,
        },
        layout_dict={"dialogue_overlay_score": 0.55, "menu_layout_score": 0.15},
    )

    assessment = scorer.assess(
        image,
        _adaptive(
            distinctiveness_score=0.68,
            information_score=0.28,
            visibility_score=0.32,
        ),
    )

    assert assessment.scene_label.value != "event"
    assert assessment.transition_suppressed_event is True
    assert assessment.bright_washout_score >= 0.52
    assert assessment.veiled_transition_score >= 0.34


def test_scene_scorer_suppresses_event0026_like_dimmed_system_frame() -> None:
    """dimmed save/title 系フレームは raw event winner でも event から外れること."""
    scorer = SceneScorer(DummyModelManager())
    image = create_analyzed_image(
        path="/tmp/event0026.jpg",
        raw_metrics_dict={
            "brightness": 78.0,
            "contrast": 6.0,
            "edge_density": 0.035,
            "luminance_entropy": 0.84,
            "luminance_range": 12.0,
            "near_black_ratio": 0.08,
            "near_white_ratio": 0.04,
            "dominant_tone_ratio": 0.80,
        },
        clip_features=torch.tensor([0.18, 0.52, 0.30]).numpy(),
        combined_features=torch.tensor([0.18, 0.52, 0.30, 0.0]).numpy(),
        normalized_metrics_dict={
            "action_intensity": 0.08,
            "ui_density": 0.40,
            "dramatic_score": 0.18,
            "color_richness": 0.18,
        },
        layout_dict={
            "menu_layout_score": 0.32,
            "title_layout_score": 0.36,
        },
    )

    assessment = scorer.assess(
        image,
        _adaptive(
            distinctiveness_score=0.60,
            information_score=0.34,
            visibility_score=0.42,
        ),
    )

    assert assessment.scene_label.value != "event"
    assert assessment.transition_suppressed_event is True
    assert assessment.veiled_transition_score >= 0.34


def test_scene_scorer_keeps_readable_event_even_when_bright() -> None:
    """明るくても readable な event は suppress されないこと."""
    scorer = SceneScorer(DummyModelManager())
    image = create_analyzed_image(
        path="/tmp/readable_bright_event.jpg",
        raw_metrics_dict={
            "brightness": 198.0,
            "contrast": 16.0,
            "edge_density": 0.16,
            "luminance_entropy": 1.25,
            "luminance_range": 36.0,
            "near_white_ratio": 0.22,
            "dominant_tone_ratio": 0.58,
        },
        clip_features=torch.tensor([0.10, 0.70, 0.05]).numpy(),
        combined_features=torch.tensor([0.10, 0.70, 0.05, 0.0]).numpy(),
        normalized_metrics_dict={
            "action_intensity": 0.18,
            "ui_density": 0.20,
            "dramatic_score": 0.84,
            "color_richness": 0.76,
        },
        layout_dict={"dialogue_overlay_score": 0.48},
    )

    assessment = scorer.assess(
        image,
        _adaptive(
            distinctiveness_score=0.72,
            information_score=0.78,
            visibility_score=0.76,
        ),
    )

    assert assessment.scene_label.value == "event"
    assert assessment.transition_suppressed_event is False
    assert assessment.veiled_transition_score < 0.34
