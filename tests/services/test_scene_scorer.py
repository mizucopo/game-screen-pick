"""SceneScorer の単体テスト."""

import numpy as np

from src.constants.scene_label import SceneLabel
from src.models.scene_mix import SceneMix
from src.services.scene_scorer import SceneScorer
from tests.conftest import create_analyzed_image


def _feature(index: int) -> np.ndarray:
    vector = np.zeros(576, dtype=np.float32)
    vector[index] = 1.0
    return vector


def _near_duplicate(base: np.ndarray, index: int) -> np.ndarray:
    feature = base.copy()
    feature[index] = 0.01
    return feature


def test_scene_scorer_assigns_dense_cluster_to_play() -> None:
    """近傍密度が高い候補群が play へ割り当てられること."""
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

    assessments = scorer.assess_batch(images, SceneMix(play=0.7, event=0.3))
    labels_by_path = {
        image.path: assessment.scene_label
        for image, assessment in zip(images, assessments, strict=True)
    }

    assert labels_by_path["/tmp/play_a.jpg"] == SceneLabel.PLAY
    assert labels_by_path["/tmp/play_b.jpg"] == SceneLabel.PLAY
    assert labels_by_path["/tmp/play_c.jpg"] == SceneLabel.PLAY
    assert labels_by_path["/tmp/event_a.jpg"] == SceneLabel.PLAY
    assert labels_by_path["/tmp/event_b.jpg"] == SceneLabel.EVENT


def test_scene_scorer_normalizes_density_scores() -> None:
    """density_score が 0..1 に正規化されること."""
    scorer = SceneScorer()
    images = [
        create_analyzed_image(path="/tmp/a.jpg", combined_features=_feature(0)),
        create_analyzed_image(path="/tmp/b.jpg", combined_features=_feature(1)),
    ]

    assessments = scorer.assess_batch(images, SceneMix(play=0.5, event=0.5))

    assert all(0.0 <= assessment.density_score <= 1.0 for assessment in assessments)
    assert {assessment.scene_label for assessment in assessments} == {
        SceneLabel.PLAY,
        SceneLabel.EVENT,
    }
