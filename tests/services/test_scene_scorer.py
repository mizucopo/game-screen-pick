"""SceneScorerの単体テスト.

CLIP 風の固定埋め込みとレイアウトヒューリスティクスを使い、
gameplay / event / other の3分類が期待通りに働くかを確認する。
"""

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


def test_scene_scorer_prefers_event_for_event_like_image() -> None:
    """イベント画面ではevent_scoreが最も高くなること.

    Given:
        - event 向け埋め込みと会話オーバーレイ寄りの特徴を持つ画像がある
    When:
        - SceneScorer で評価する
    Then:
        - scene label が event になり、他ラベルより高得点になること
    """
    # Arrange
    scorer = SceneScorer(DummyModelManager())
    image = create_analyzed_image(
        path="/tmp/event.jpg",
        clip_features=torch.tensor([0.0, 1.0, 0.0]).numpy(),
        combined_features=torch.tensor([0.0, 1.0, 0.0, 0.0]).numpy(),
        layout_dict={"dialogue_overlay_score": 0.6, "menu_layout_score": 0.0},
    )

    # Act
    assessment = scorer.assess(image)

    # Assert
    assert assessment.scene_label.value == "event"
    assert assessment.event_score > assessment.gameplay_score
    assert assessment.event_score > assessment.other_score


def test_scene_scorer_prefers_other_for_other_like_image() -> None:
    """メニュー系画面ではother_scoreが最も高くなること.

    Given:
        - other 向け埋め込みとメニュー / タイトル寄りの特徴を持つ画像がある
    When:
        - SceneScorer で評価する
    Then:
        - scene label が other になり、他ラベルより高得点になること
    """
    # Arrange
    scorer = SceneScorer(DummyModelManager())
    image = create_analyzed_image(
        path="/tmp/other.jpg",
        clip_features=torch.tensor([0.0, 0.0, 1.0]).numpy(),
        combined_features=torch.tensor([0.0, 0.0, 1.0, 0.0]).numpy(),
        layout_dict={"menu_layout_score": 0.6, "title_layout_score": 0.4},
    )

    # Act
    assessment = scorer.assess(image)

    # Assert
    assert assessment.scene_label.value == "other"
    assert assessment.other_score > assessment.gameplay_score
    assert assessment.other_score > assessment.event_score
