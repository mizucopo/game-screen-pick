"""SceneScorer の単体テスト."""

import numpy as np

from src.constants.scene_label import SceneLabel
from src.models.scene_mix import SceneMix
from src.services.scene_scorer import SceneScorer
from tests.conftest import _feature, _near_duplicate, create_analyzed_image


def test_scene_scorer_assigns_dense_cluster_to_play() -> None:
    """近傍密度が高い候補群が play へ割り当てられること.

    Arrange:
        - 互いに似た特徴を持つ画像群（高密度クラスタ）がある
        - 孤立した特徴を持つ画像群（低密度）がある
        - scene_mix比率が70/30に設定されている
    Act:
        - SceneScorerでscene評価を行う
    Assert:
        - 高密度クラスタの画像がplayに割り当てられること
        - 低密度の画像の一部がeventに割り当てられること
    """
    # Arrange
    scorer = SceneScorer()
    base = _feature(0)
    images = [
        create_analyzed_image(
            path="/tmp/play_a.jpg",
            combined_features=base,
        ),
        create_analyzed_image(
            path="/tmp/play_b.jpg",
            combined_features=_near_duplicate(base, 1),
        ),
        create_analyzed_image(
            path="/tmp/play_c.jpg",
            combined_features=_near_duplicate(base, 2),
        ),
        create_analyzed_image(
            path="/tmp/event_a.jpg",
            combined_features=_feature(100),
        ),
        create_analyzed_image(
            path="/tmp/event_b.jpg",
            combined_features=_feature(200),
        ),
    ]

    # Act
    assessments = scorer.assess_batch(images, SceneMix(play=0.7, event=0.3))
    labels_by_path = {
        image.path: assessment.scene_label
        for image, assessment in zip(images, assessments, strict=True)
    }

    # Assert
    assert labels_by_path["/tmp/play_a.jpg"] == SceneLabel.PLAY
    assert labels_by_path["/tmp/play_b.jpg"] == SceneLabel.PLAY
    assert labels_by_path["/tmp/play_c.jpg"] == SceneLabel.PLAY
    assert labels_by_path["/tmp/event_a.jpg"] == SceneLabel.PLAY
    assert labels_by_path["/tmp/event_b.jpg"] == SceneLabel.EVENT


def test_scene_scorer_normalizes_density_scores() -> None:
    """density_score が 0..1 に正規化されること.

    Arrange:
        - 異なる特徴を持つ2つの画像がある
        - scene_mix比率が50/50に設定されている
    Act:
        - SceneScorerでscene評価を行う
    Assert:
        - すべてのdensity_scoreが0.0〜1.0の範囲になること
        - play/event両方のラベルが割り当てられること
    """
    # Arrange
    scorer = SceneScorer()
    images = [
        create_analyzed_image(path="/tmp/a.jpg", combined_features=_feature(0)),
        create_analyzed_image(path="/tmp/b.jpg", combined_features=_feature(1)),
    ]

    # Act
    assessments = scorer.assess_batch(images, SceneMix(play=0.5, event=0.5))

    # Assert
    assert all(0.0 <= assessment.density_score <= 1.0 for assessment in assessments)
    assert {assessment.scene_label for assessment in assessments} == {
        SceneLabel.PLAY,
        SceneLabel.EVENT,
    }


def test_scene_scorer_handles_zero_norm_features() -> None:
    """ゼロベクトルを含む画像群でもdensity_scoreが0.0〜1.0に収まること.

    Arrange:
        - ゼロベクトルの特徴を持つ画像が含まれている
        - 有効な特徴を持つ画像も含まれている
    Act:
        - SceneScorerでscene評価を行う
    Assert:
        - すべてのdensity_scoreが0.0〜1.0の範囲になること
        - NaNが含まれないこと
    """
    # Arrange
    scorer = SceneScorer()
    images = [
        create_analyzed_image(
            path="/tmp/zero.jpg",
            combined_features=np.zeros(576, dtype=np.float32),
        ),
        create_analyzed_image(path="/tmp/a.jpg", combined_features=_feature(0)),
        create_analyzed_image(path="/tmp/b.jpg", combined_features=_feature(1)),
    ]

    # Act
    assessments = scorer.assess_batch(images, SceneMix(play=0.7, event=0.3))

    # Assert
    for assessment in assessments:
        assert 0.0 <= assessment.density_score <= 1.0
        assert not np.isnan(assessment.density_score)
