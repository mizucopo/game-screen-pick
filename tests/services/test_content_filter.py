"""ContentFilterの単体テスト."""

import numpy as np
import torch

from src.services.content_filter import ContentFilter
from src.services.whole_input_profiler import WholeInputProfiler
from tests.conftest import create_analyzed_image


def _feature(index: int, delta_index: int | None = None, delta: float = 0.0) -> np.ndarray:
    """content filter 用の決定的な特徴ベクトルを作る."""
    feature = np.zeros(101, dtype=np.float32)
    feature[index] = 1.0
    if delta_index is not None:
        feature[delta_index] = delta
    return feature


def test_content_filter_rejects_flat_frames_and_keeps_informative_dark_frames() -> None:
    """暗いタイトル群でも高情報量フレームだけが残ること.

    Given:
        - blackout、whiteout、単色、フェード遷移などの低情報量フレームを含む画像群がある
        - 高情報量の暗いgameplay/eventフレームもある
    When:
        - ContentFilterでフィルタリングする
    Then:
        - 低情報量フレームが除外され、高情報量フレームのみ残ること
    """
    # Arrange
    dark_gameplay = create_analyzed_image(
        path="/tmp/dark_gameplay.jpg",
        raw_metrics_dict={
            "brightness": 28.0,
            "contrast": 22.0,
            "edge_density": 0.34,
            "action_intensity": 24.0,
            "luminance_entropy": 1.5,
            "luminance_range": 48.0,
            "near_black_ratio": 0.12,
            "near_white_ratio": 0.0,
            "dominant_tone_ratio": 0.62,
        },
        clip_features=torch.tensor([1.0, 0.0, 0.0]).numpy(),
        combined_features=np.pad(_feature(0), (0, 475)),
        content_features=_feature(0),
    )
    dark_event = create_analyzed_image(
        path="/tmp/dark_event.jpg",
        raw_metrics_dict={
            "brightness": 26.0,
            "contrast": 18.0,
            "edge_density": 0.28,
            "action_intensity": 20.0,
            "luminance_entropy": 1.2,
            "luminance_range": 40.0,
            "near_black_ratio": 0.18,
            "near_white_ratio": 0.0,
            "dominant_tone_ratio": 0.67,
        },
        clip_features=torch.tensor([0.0, 1.0, 0.0]).numpy(),
        combined_features=np.pad(_feature(1), (0, 475)),
        content_features=_feature(1),
        layout_dict={"dialogue_overlay_score": 0.6},
    )
    blackout = create_analyzed_image(
        path="/tmp/blackout.jpg",
        raw_metrics_dict={
            "brightness": 1.0,
            "contrast": 0.2,
            "edge_density": 0.001,
            "action_intensity": 0.1,
            "luminance_entropy": 0.05,
            "luminance_range": 1.0,
            "near_black_ratio": 0.99,
            "near_white_ratio": 0.0,
            "dominant_tone_ratio": 1.0,
        },
        combined_features=np.pad(_feature(2), (0, 475)),
        content_features=_feature(2),
    )
    whiteout = create_analyzed_image(
        path="/tmp/whiteout.jpg",
        raw_metrics_dict={
            "brightness": 250.0,
            "contrast": 0.3,
            "edge_density": 0.001,
            "action_intensity": 0.1,
            "luminance_entropy": 0.05,
            "luminance_range": 1.0,
            "near_black_ratio": 0.0,
            "near_white_ratio": 0.99,
            "dominant_tone_ratio": 1.0,
        },
        combined_features=np.pad(_feature(2, 3, 0.01), (0, 475)),
        content_features=_feature(2, 3, 0.01),
    )
    single_tone = create_analyzed_image(
        path="/tmp/single_tone.jpg",
        raw_metrics_dict={
            "brightness": 120.0,
            "contrast": 0.2,
            "edge_density": 0.001,
            "action_intensity": 0.1,
            "luminance_entropy": 0.08,
            "luminance_range": 3.0,
            "near_black_ratio": 0.0,
            "near_white_ratio": 0.0,
            "dominant_tone_ratio": 0.96,
        },
        combined_features=np.pad(_feature(2, 4, 0.02), (0, 475)),
        content_features=_feature(2, 4, 0.02),
    )
    fade_transition = create_analyzed_image(
        path="/tmp/fade_transition.jpg",
        raw_metrics_dict={
            "brightness": 12.0,
            "contrast": 0.0,
            "edge_density": 0.0,
            "action_intensity": 0.0,
            "luminance_entropy": 0.0,
            "luminance_range": 0.0,
            "near_black_ratio": 0.85,
            "near_white_ratio": 0.0,
            "dominant_tone_ratio": 0.90,
        },
        combined_features=np.pad(_feature(2, 5, 0.005), (0, 475)),
        content_features=_feature(2, 5, 0.005),
    )
    images = [
        dark_gameplay,
        dark_event,
        blackout,
        whiteout,
        single_tone,
        fade_transition,
    ]
    content_filter = ContentFilter(WholeInputProfiler())

    # Act
    result = content_filter.filter(images)

    # Assert
    assert {image.path for image in result.kept_images} == {
        "/tmp/dark_gameplay.jpg",
        "/tmp/dark_event.jpg",
    }
    assert result.rejected_by_content_filter == 4
    assert result.content_filter_breakdown == {
        "blackout": 1,
        "whiteout": 1,
        "single_tone": 1,
        "fade_transition": 1,
    }
    assert (
        result.adaptive_scores_by_image_id[id(dark_gameplay)].information_score
        > result.adaptive_scores_by_image_id[id(fade_transition)].information_score
    )
