"""SceneScorerの単体テスト.

CLIP 風の固定埋め込みとレイアウトヒューリスティクスを使い、
gameplay / event / other の3分類が期待通りに働くかを確認する。
"""

import pytest
import torch

from src.services.scene_scorer import SceneScorer
from tests.conftest import create_analyzed_image
from tests.dummy_model_manager import DummyModelManager


def test_scene_scorer_prefers_gameplay_for_gameplay_like_image() -> None:
    """通常画面ではgameplay_scoreが最も高くなること.

    Given:
        - gameplay 向け埋め込みと HUD 寄りの特徴を持つ画像がある
    When:
        - SceneScorer で評価する
    Then:
        - scene label が gameplay になり、他ラベルより高得点になること
    """
    # Arrange
    scorer = SceneScorer(DummyModelManager())
    image = create_analyzed_image(
        path="/tmp/gameplay.jpg",
        clip_features=torch.tensor([1.0, 0.0, 0.0]).numpy(),
        combined_features=torch.tensor([1.0, 0.0, 0.0, 0.0]).numpy(),
        layout_dict={"dialogue_overlay_score": 0.0, "menu_layout_score": 0.0},
    )

    # Act
    assessment = scorer.assess(image)

    # Assert
    assert assessment.scene_label.value == "gameplay"
    assert assessment.gameplay_score > assessment.event_score
    assert assessment.gameplay_score > assessment.other_score


def test_scene_scorer_rewards_frequent_gameplay_cluster() -> None:
    """頻出クラスタに属する画像ほどgameplayへ寄ること.

    Given:
        - gameplay向け埋め込みを持つ同種の画像がある
    When:
        - 差分量スコアの低い場合と高い場合で評価する
    Then:
        - 頻出側の方が gameplay_score が高くなること
    """
    # Arrange
    scorer = SceneScorer(DummyModelManager())
    image = create_analyzed_image(
        path="/tmp/gameplay_cluster.jpg",
        clip_features=torch.tensor([0.6, 0.0, 0.0]).numpy(),
        combined_features=torch.tensor([0.6, 0.0, 0.0, 0.0]).numpy(),
    )

    # Act
    frequent = scorer.assess(image, distinctiveness_score=0.1)
    rare = scorer.assess(image, distinctiveness_score=0.9)

    # Assert
    assert frequent.scene_label.value == "gameplay"
    assert frequent.gameplay_score > frequent.other_score
    assert frequent.gameplay_score > rare.gameplay_score


def test_scene_scorer_prefers_event_for_boss_intro_like_image() -> None:
    """会話なしの演出シーンでもeventへ寄ること.

    Given:
        - event 向け埋め込みとドラマ性の高い導入演出画面がある
    When:
        - SceneScorer で評価する
    Then:
        - scene label が event になり、他ラベルより高得点になること
    """
    # Arrange
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

    # Act
    assessment = scorer.assess(image)

    # Assert
    assert assessment.scene_label.value == "event"
    assert assessment.event_score > assessment.gameplay_score
    assert assessment.event_score > assessment.other_score


def test_scene_scorer_keeps_event_when_event_raw_winner_is_close_to_other() -> None:
    """event が生スコア最大なら other fallback で潰れないこと."""
    # Arrange
    scorer = SceneScorer(DummyModelManager())
    image = create_analyzed_image(
        path="/tmp/event_close_to_other.jpg",
        clip_features=torch.tensor([0.0, 0.36, 0.35]).numpy(),
        combined_features=torch.tensor([0.0, 0.36, 0.35, 0.0]).numpy(),
        normalized_metrics_dict={
            "action_intensity": 0.4,
            "ui_density": 0.6,
            "dramatic_score": 0.2,
            "color_richness": 0.2,
        },
        layout_dict={"dialogue_overlay_score": 0.2},
    )

    # Act
    assessment = scorer.assess(image, distinctiveness_score=0.3)

    # Assert
    assert assessment.event_score > assessment.other_score
    assert assessment.event_score - assessment.other_score <= 0.05
    assert assessment.scene_label.value == "event"


def test_scene_scorer_promotes_gameplay_near_miss_to_event() -> None:
    """演出寄りの gameplay near miss だけ event へ昇格すること."""
    # Arrange
    scorer = SceneScorer(DummyModelManager())
    image = create_analyzed_image(
        path="/tmp/gameplay_near_miss.jpg",
        clip_features=torch.tensor([0.436, 0.20, 0.14]).numpy(),
        combined_features=torch.tensor([0.436, 0.20, 0.14, 0.0]).numpy(),
        normalized_metrics_dict={
            "action_intensity": 0.35,
            "ui_density": 0.35,
            "dramatic_score": 0.8,
            "color_richness": 0.5,
        },
        layout_dict={"dialogue_overlay_score": 0.0, "menu_layout_score": 0.0},
    )

    # Act
    assessment = scorer.assess(image, distinctiveness_score=0.75)

    # Assert
    assert assessment.gameplay_score > assessment.event_score
    assert assessment.gameplay_score - assessment.event_score <= 0.01
    assert assessment.event_score >= 0.42
    assert assessment.other_score <= assessment.event_score - 0.01
    assert assessment.scene_label.value == "event"
    assert assessment.event_score >= assessment.other_score


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
    # Arrange
    scorer = SceneScorer(DummyModelManager())
    image = create_analyzed_image(
        path="/tmp/gameplay_not_promoted.jpg",
        clip_features=torch.tensor([0.436, 0.20, 0.14]).numpy(),
        combined_features=torch.tensor([0.436, 0.20, 0.14, 0.0]).numpy(),
        normalized_metrics_dict=normalized_metrics_dict,
        layout_dict={"dialogue_overlay_score": 0.0, "menu_layout_score": 0.0},
    )

    # Act
    assessment = scorer.assess(image, distinctiveness_score=distinctiveness_score)

    # Assert
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
    """補助UIや遷移画面はotherへ寄ること.

    Given:
        - map、equipment、shop、result_reward などの補助UI画面がある
        - それぞれ menu_layout_score や title_layout_score を持つ
    When:
        - SceneScorer で評価する
    Then:
        - scene label が other になり、他ラベルより高得点になること
    """
    # Arrange
    scorer = SceneScorer(DummyModelManager())
    image = create_analyzed_image(
        path=f"/tmp/{screen_name}.jpg",
        clip_features=torch.tensor([0.0, 0.0, 1.0]).numpy(),
        combined_features=torch.tensor([0.0, 0.0, 1.0, 0.0]).numpy(),
        normalized_metrics_dict={"action_intensity": 0.1, "ui_density": 0.9},
        layout_dict=layout_dict,
    )

    # Act
    assessment = scorer.assess(image, distinctiveness_score=0.8)

    # Assert
    assert assessment.scene_label.value == "other"
    assert assessment.other_score > assessment.gameplay_score
    assert assessment.other_score > assessment.event_score


def test_scene_scorer_keeps_support_ui_near_miss_as_other() -> None:
    """support UI が強い other near miss は event に昇格しないこと."""
    # Arrange
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

    # Act
    assessment = scorer.assess(image, distinctiveness_score=0.7)

    # Assert
    assert assessment.other_score > assessment.event_score
    assert assessment.other_score - assessment.event_score <= 0.05
    assert assessment.scene_label.value == "other"


def test_scene_scorer_falls_back_to_other_for_ambiguous_gameplay_and_other() -> None:
    """gameplay と other が僅差なら other に倒すこと.

    Given:
        - gameplay と other のスコアが僅差（0.05以下）の画像がある
    When:
        - SceneScorer で評価する
    Then:
        - scene label が other になること
        - confidence が 0.0 になること
    """
    # Arrange
    scorer = SceneScorer(DummyModelManager())
    image = create_analyzed_image(
        path="/tmp/ambiguous.jpg",
        clip_features=torch.tensor([0.73, 0.0, 0.65]).numpy(),
        combined_features=torch.tensor([0.73, 0.0, 0.65, 0.0]).numpy(),
    )

    # Act
    assessment = scorer.assess(image, distinctiveness_score=0.5)

    # Assert
    assert assessment.gameplay_score > assessment.other_score
    assert assessment.gameplay_score - assessment.other_score <= 0.05
    assert assessment.scene_label.value == "other"
    assert assessment.scene_confidence == 0.0


def test_scene_scorer_uses_neutral_distinctiveness_when_omitted() -> None:
    """distinctiveness未指定時は中立値0.5と同じ判定になること.

    Given:
        - distinctiveness_score を指定せずに評価する画像がある
    When:
        - 明示的に0.5を指定した場合と比較する
    Then:
        - 両者の結果が完全に一致すること
    """
    # Arrange
    scorer = SceneScorer(DummyModelManager())
    image = create_analyzed_image(
        path="/tmp/neutral_default.jpg",
        clip_features=torch.tensor([0.0, 1.0, 0.0]).numpy(),
        combined_features=torch.tensor([0.0, 1.0, 0.0, 0.0]).numpy(),
        normalized_metrics_dict={"dramatic_score": 0.8, "ui_density": 0.2},
    )

    # Act
    without_arg = scorer.assess(image)
    explicit_neutral = scorer.assess(image, distinctiveness_score=0.5)

    # Assert
    assert without_arg == explicit_neutral
