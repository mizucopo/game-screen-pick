"""GameScreenPickerの単体テスト."""

import numpy as np

from src.models.scene_mix import SceneMix
from src.models.selection_config import SelectionConfig
from src.services.game_screen_picker import GameScreenPicker
from tests.conftest import create_analyzed_image
from tests.fake_analyzer import FakeAnalyzer


def _feature(index: int) -> np.ndarray:
    feature = np.zeros(576, dtype=np.float32)
    feature[index] = 1.0
    return feature


def _near_duplicate(base: np.ndarray, index: int) -> np.ndarray:
    feature = base.copy()
    feature[index] = 0.01
    return feature


def test_select_from_analyzed_filters_content_before_play_event_assignment() -> None:
    """content filter の除外が先に適用されること."""
    # Arrange
    dark = create_analyzed_image(
        path="/tmp/dark.jpg",
        raw_metrics_dict={
            "near_black_ratio": 0.98,
            "luminance_entropy": 0.2,
            "luminance_range": 10.0,
        },
        combined_features=_feature(0),
    )
    play = create_analyzed_image(
        path="/tmp/play.jpg",
        combined_features=_feature(1),
    )
    event = create_analyzed_image(
        path="/tmp/event.jpg",
        combined_features=_feature(100),
    )
    analyzed_images = [dark, play, event]
    picker = GameScreenPicker(
        analyzer=FakeAnalyzer(analyzed_images),
        config=SelectionConfig(scene_mix=SceneMix(play=0.5, event=0.5)),
    )

    # Act
    selected, rejected, stats = picker.select_from_analyzed(analyzed_images, num=2)

    # Assert
    assert {candidate.path for candidate in selected} == {"/tmp/play.jpg", "/tmp/event.jpg"}
    assert rejected == []
    assert stats.rejected_by_content_filter == 1
    assert stats.content_filter_breakdown["blackout"] == 1
    assert stats.scene_mix_actual == {"play": 1, "event": 1}


def test_select_from_analyzed_assigns_dense_candidates_to_play() -> None:
    """密度の高いクラスタが play へ寄ること."""
    # Arrange
    base = _feature(0)
    analyzed_images = [
        create_analyzed_image(
            path="/tmp/play_0.jpg",
            combined_features=base,
        ),
        create_analyzed_image(
            path="/tmp/play_1.jpg",
            combined_features=_near_duplicate(base, 1),
        ),
        create_analyzed_image(
            path="/tmp/play_2.jpg",
            combined_features=_near_duplicate(base, 2),
        ),
        create_analyzed_image(
            path="/tmp/event_0.jpg",
            combined_features=_feature(100),
        ),
        create_analyzed_image(
            path="/tmp/event_1.jpg",
            combined_features=_feature(200),
        ),
    ]
    picker = GameScreenPicker(
        analyzer=FakeAnalyzer(analyzed_images),
        config=SelectionConfig(scene_mix=SceneMix(play=0.7, event=0.3)),
    )

    # Act
    selected, rejected, stats = picker.select_from_analyzed(analyzed_images, num=5)

    # Assert
    assert len(selected) >= 2
    assert all(candidate.scene_assessment.scene_label.value in {"play", "event"} for candidate in selected)
    assert len(rejected) >= 1
    assert stats.scene_distribution == {"play": 4, "event": 1}
    assert stats.scene_mix_target == {"play": 4, "event": 1}
    assert stats.scene_mix_actual["play"] >= 1
    assert stats.scene_mix_actual["event"] >= 1


def test_select_from_analyzed_spreads_score_bands_and_rejects_duplicates() -> None:
    """band 分散と global 類似度除外が同時に働くこと."""
    # Arrange
    base = _feature(0)
    analyzed_images = [
        create_analyzed_image(path="/tmp/play_low.jpg", combined_features=_feature(10)),
        create_analyzed_image(path="/tmp/play_mid.jpg", combined_features=_feature(11)),
        create_analyzed_image(path="/tmp/play_high.jpg", combined_features=base),
        create_analyzed_image(
            path="/tmp/event_dup.jpg",
            combined_features=_near_duplicate(base, 1),
        ),
        create_analyzed_image(path="/tmp/event_far.jpg", combined_features=_feature(100)),
        create_analyzed_image(path="/tmp/event_far2.jpg", combined_features=_feature(200)),
    ]
    picker = GameScreenPicker(
        analyzer=FakeAnalyzer(analyzed_images),
        config=SelectionConfig(scene_mix=SceneMix(play=0.5, event=0.5)),
    )

    # Act
    selected, _rejected, stats = picker.select_from_analyzed(analyzed_images, num=4)

    # Assert
    assert stats.rejected_by_similarity >= 1
    assert len({candidate.score_band for candidate in selected}) >= 2
    assert all(candidate.score_band is not None for candidate in selected)
