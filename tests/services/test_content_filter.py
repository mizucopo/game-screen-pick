"""ContentFilterの単体テスト."""

import tempfile
from pathlib import Path

import numpy as np
import torch

from src.services.content_filter import ContentFilter
from src.services.game_screen_picker import GameScreenPicker
from src.services.whole_input_profiler import WholeInputProfiler
from tests.conftest import create_analyzed_image


def _feature(
    index: int,
    delta_index: int | None = None,
    delta: float = 0.0,
) -> np.ndarray:
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
        "temporal_transition": 0,
    }
    assert result.whole_input_profile.brightness.p50 > 0.0
    assert result.whole_input_profile.near_white_ratio.p90 >= 0.0
    assert (
        result.adaptive_scores_by_image_id[id(dark_gameplay)].information_score
        > result.adaptive_scores_by_image_id[id(fade_transition)].information_score
    )
    assert (
        result.adaptive_scores_by_image_id[id(dark_gameplay)].visibility_score
        > result.adaptive_scores_by_image_id[id(fade_transition)].visibility_score
    )


def test_content_filter_rejects_mid_fade_frames_with_70_percent_dark_ratio() -> None:
    """70〜75% 暗転途中フレームが static fade 判定で落ちること.

    Given:
        - 70%と75%の暗転途中フレームを含む画像群がある
        - 通常の高情報量フレームもある
    When:
        - ContentFilterでフィルタリングする
    Then:
        - 暗転途中フレームがfade_transitionとして除外されること
    """
    # Arrange
    images = [
        create_analyzed_image(
            path="/tmp/good1.jpg",
            raw_metrics_dict={
                "contrast": 18.0,
                "edge_density": 0.28,
                "action_intensity": 20.0,
                "luminance_entropy": 1.2,
                "luminance_range": 40.0,
                "near_black_ratio": 0.18,
                "dominant_tone_ratio": 0.67,
            },
            content_features=_feature(0),
            combined_features=np.pad(_feature(0), (0, 475)),
        ),
        create_analyzed_image(
            path="/tmp/good2.jpg",
            raw_metrics_dict={
                "contrast": 16.0,
                "edge_density": 0.22,
                "action_intensity": 17.0,
                "luminance_entropy": 1.0,
                "luminance_range": 34.0,
                "near_black_ratio": 0.22,
                "dominant_tone_ratio": 0.70,
            },
            content_features=_feature(1),
            combined_features=np.pad(_feature(1), (0, 475)),
        ),
        create_analyzed_image(
            path="/tmp/fade70.jpg",
            raw_metrics_dict={
                "contrast": 3.5,
                "edge_density": 0.015,
                "action_intensity": 1.2,
                "luminance_entropy": 0.42,
                "luminance_range": 9.0,
                "near_black_ratio": 0.70,
                "dominant_tone_ratio": 0.88,
            },
            content_features=_feature(2),
            combined_features=np.pad(_feature(2), (0, 475)),
        ),
        create_analyzed_image(
            path="/tmp/fade75.jpg",
            raw_metrics_dict={
                "contrast": 2.5,
                "edge_density": 0.010,
                "action_intensity": 0.8,
                "luminance_entropy": 0.36,
                "luminance_range": 8.0,
                "near_black_ratio": 0.75,
                "dominant_tone_ratio": 0.89,
            },
            content_features=_feature(3),
            combined_features=np.pad(_feature(3), (0, 475)),
        ),
    ]

    # Act
    result = ContentFilter(WholeInputProfiler()).filter(images)

    # Assert
    assert {image.path for image in result.kept_images} == {
        "/tmp/good1.jpg",
        "/tmp/good2.jpg",
    }
    assert result.content_filter_breakdown["fade_transition"] == 2
    assert result.content_filter_breakdown["temporal_transition"] == 0


def test_content_filter_rejects_borderline_whiteout_frame() -> None:
    """90%台前半の白飛びフレームも whiteout で除外されること."""
    images = [
        create_analyzed_image(
            path="/tmp/good_event.jpg",
            raw_metrics_dict={
                "brightness": 168.0,
                "contrast": 18.0,
                "edge_density": 0.20,
                "action_intensity": 14.0,
                "luminance_entropy": 1.3,
                "luminance_range": 36.0,
                "near_white_ratio": 0.20,
                "dominant_tone_ratio": 0.62,
            },
            content_features=_feature(0),
            combined_features=np.pad(_feature(0), (0, 475)),
        ),
        create_analyzed_image(
            path="/tmp/borderline_whiteout.jpg",
            raw_metrics_dict={
                "brightness": 246.0,
                "contrast": 4.5,
                "edge_density": 0.01,
                "action_intensity": 0.5,
                "luminance_entropy": 0.50,
                "luminance_range": 8.0,
                "near_white_ratio": 0.93,
                "dominant_tone_ratio": 0.91,
            },
            content_features=_feature(1),
            combined_features=np.pad(_feature(1), (0, 475)),
        ),
    ]

    result = ContentFilter(WholeInputProfiler()).filter(images)

    assert {image.path for image in result.kept_images} == {"/tmp/good_event.jpg"}
    assert result.content_filter_breakdown["whiteout"] == 1


def test_content_filter_rejects_bright_washed_out_frame_as_whiteout() -> None:
    """near_white が低めでも強い白飛びは whiteout になること."""
    images = [
        create_analyzed_image(
            path="/tmp/good_event.jpg",
            raw_metrics_dict={
                "brightness": 165.0,
                "contrast": 18.0,
                "edge_density": 0.20,
                "action_intensity": 12.0,
                "luminance_entropy": 1.3,
                "luminance_range": 36.0,
                "near_white_ratio": 0.18,
                "dominant_tone_ratio": 0.58,
            },
            content_features=_feature(0),
            combined_features=np.pad(_feature(0), (0, 475)),
        ),
        create_analyzed_image(
            path="/tmp/bright_whiteout.jpg",
            raw_metrics_dict={
                "brightness": 230.0,
                "contrast": 3.0,
                "edge_density": 0.008,
                "action_intensity": 0.8,
                "luminance_entropy": 0.40,
                "luminance_range": 6.0,
                "near_white_ratio": 0.58,
                "dominant_tone_ratio": 0.93,
            },
            content_features=_feature(1),
            combined_features=np.pad(_feature(1), (0, 475)),
        ),
    ]

    result = ContentFilter(WholeInputProfiler()).filter(images)

    assert {image.path for image in result.kept_images} == {"/tmp/good_event.jpg"}
    assert result.content_filter_breakdown["whiteout"] == 1


def test_content_filter_rejects_washed_out_gameplay_frame_as_fade_transition() -> None:
    """構図が少し見える明転中 gameplay は fade_transition になること."""
    images = [
        create_analyzed_image(
            path="/tmp/good_gameplay.jpg",
            raw_metrics_dict={
                "brightness": 110.0,
                "contrast": 16.0,
                "edge_density": 0.18,
                "action_intensity": 14.0,
                "luminance_entropy": 1.2,
                "luminance_range": 34.0,
                "near_white_ratio": 0.10,
                "dominant_tone_ratio": 0.60,
            },
            content_features=_feature(0),
            combined_features=np.pad(_feature(0), (0, 475)),
        ),
        create_analyzed_image(
            path="/tmp/washed_out_gameplay.jpg",
            raw_metrics_dict={
                "brightness": 206.0,
                "contrast": 5.5,
                "edge_density": 0.025,
                "action_intensity": 2.5,
                "luminance_entropy": 0.62,
                "luminance_range": 10.0,
                "near_white_ratio": 0.36,
                "dominant_tone_ratio": 0.86,
            },
            content_features=_feature(1),
            combined_features=np.pad(_feature(1), (0, 475)),
        ),
    ]

    result = ContentFilter(WholeInputProfiler()).filter(images)

    assert {image.path for image in result.kept_images} == {"/tmp/good_gameplay.jpg"}
    assert result.content_filter_breakdown["fade_transition"] == 1


def test_content_filter_rejects_bright_and_dim_transition_frames() -> None:
    """washed-out / dimmed な遷移フレームも static fade 判定で落ちること."""
    images = [
        create_analyzed_image(
            path="/tmp/good_event.jpg",
            raw_metrics_dict={
                "brightness": 148.0,
                "contrast": 18.0,
                "edge_density": 0.20,
                "action_intensity": 12.0,
                "luminance_entropy": 1.4,
                "luminance_range": 39.0,
                "near_white_ratio": 0.08,
                "dominant_tone_ratio": 0.60,
            },
            content_features=_feature(0),
            combined_features=np.pad(_feature(0), (0, 475)),
        ),
        create_analyzed_image(
            path="/tmp/good_gameplay.jpg",
            raw_metrics_dict={
                "brightness": 92.0,
                "contrast": 17.0,
                "edge_density": 0.21,
                "action_intensity": 16.0,
                "luminance_entropy": 1.2,
                "luminance_range": 35.0,
                "near_black_ratio": 0.14,
                "dominant_tone_ratio": 0.62,
            },
            content_features=_feature(1),
            combined_features=np.pad(_feature(1), (0, 475)),
        ),
        create_analyzed_image(
            path="/tmp/bright_transition.jpg",
            raw_metrics_dict={
                "brightness": 222.0,
                "contrast": 4.0,
                "edge_density": 0.02,
                "action_intensity": 2.0,
                "luminance_entropy": 0.55,
                "luminance_range": 8.0,
                "near_white_ratio": 0.52,
                "dominant_tone_ratio": 0.88,
            },
            content_features=_feature(2),
            combined_features=np.pad(_feature(2), (0, 475)),
        ),
        create_analyzed_image(
            path="/tmp/dim_transition.jpg",
            raw_metrics_dict={
                "brightness": 36.0,
                "contrast": 3.5,
                "edge_density": 0.02,
                "action_intensity": 1.8,
                "luminance_entropy": 0.55,
                "luminance_range": 8.0,
                "near_black_ratio": 0.55,
                "dominant_tone_ratio": 0.85,
            },
            content_features=_feature(3),
            combined_features=np.pad(_feature(3), (0, 475)),
        ),
    ]

    result = ContentFilter(WholeInputProfiler()).filter(images)

    assert {image.path for image in result.kept_images} == {
        "/tmp/good_event.jpg",
        "/tmp/good_gameplay.jpg",
    }
    assert result.rejected_by_content_filter == 2
    assert (
        result.content_filter_breakdown["whiteout"]
        + result.content_filter_breakdown["fade_transition"]
        == 2
    )


def test_content_filter_rejects_event0026_like_dimmed_system_frame() -> None:
    """dimmed title/save 系フレームが fade_transition で落ちること."""
    images = [
        create_analyzed_image(
            path="/tmp/good_event.jpg",
            raw_metrics_dict={
                "brightness": 148.0,
                "contrast": 18.0,
                "edge_density": 0.20,
                "action_intensity": 12.0,
                "luminance_entropy": 1.4,
                "luminance_range": 39.0,
                "near_white_ratio": 0.08,
                "dominant_tone_ratio": 0.60,
            },
            content_features=_feature(0),
            combined_features=np.pad(_feature(0), (0, 475)),
        ),
        create_analyzed_image(
            path="/tmp/event0026.jpg",
            raw_metrics_dict={
                "brightness": 74.0,
                "contrast": 6.0,
                "edge_density": 0.035,
                "action_intensity": 1.5,
                "luminance_entropy": 0.82,
                "luminance_range": 12.0,
                "near_black_ratio": 0.08,
                "near_white_ratio": 0.04,
                "dominant_tone_ratio": 0.82,
            },
            normalized_metrics_dict={
                "action_intensity": 0.08,
                "ui_density": 0.38,
            },
            layout_dict={
                "menu_layout_score": 0.36,
                "title_layout_score": 0.42,
            },
            content_features=_feature(1),
            combined_features=np.pad(_feature(1), (0, 475)),
        ),
    ]

    result = ContentFilter(WholeInputProfiler()).filter(images)

    assert {image.path for image in result.kept_images} == {"/tmp/good_event.jpg"}
    assert result.content_filter_breakdown["fade_transition"] == 1


def test_content_filter_rejects_event0031_like_dimmed_gameplay_frame() -> None:
    """暗い veiled gameplay も fade_transition で落ちること."""
    images = [
        create_analyzed_image(
            path="/tmp/good_gameplay.jpg",
            raw_metrics_dict={
                "brightness": 110.0,
                "contrast": 16.0,
                "edge_density": 0.20,
                "action_intensity": 15.0,
                "luminance_entropy": 1.2,
                "luminance_range": 34.0,
                "near_black_ratio": 0.12,
                "dominant_tone_ratio": 0.58,
            },
            content_features=_feature(0),
            combined_features=np.pad(_feature(0), (0, 475)),
        ),
        create_analyzed_image(
            path="/tmp/event0031.jpg",
            raw_metrics_dict={
                "brightness": 46.0,
                "contrast": 5.5,
                "edge_density": 0.03,
                "action_intensity": 2.0,
                "luminance_entropy": 0.70,
                "luminance_range": 10.0,
                "near_black_ratio": 0.30,
                "near_white_ratio": 0.0,
                "dominant_tone_ratio": 0.80,
            },
            normalized_metrics_dict={
                "action_intensity": 0.10,
                "ui_density": 0.28,
            },
            content_features=_feature(1),
            combined_features=np.pad(_feature(1), (0, 475)),
        ),
    ]

    result = ContentFilter(WholeInputProfiler()).filter(images)

    assert {image.path for image in result.kept_images} == {"/tmp/good_gameplay.jpg"}
    assert result.content_filter_breakdown["fade_transition"] == 1


def test_relative_transition_uses_whole_input_brightness_tendency() -> None:
    """入力全体の通常明度帯から外れた bright/dark outlier を落とすこと."""
    images = [
        create_analyzed_image(
            path=f"/tmp/good_{index}.jpg",
            raw_metrics_dict={
                "brightness": 102.0 + index * 6.0,
                "contrast": 16.0,
                "edge_density": 0.17,
                "action_intensity": 12.0,
                "luminance_entropy": 1.15,
                "luminance_range": 33.0,
                "near_black_ratio": 0.05,
                "near_white_ratio": 0.05,
                "dominant_tone_ratio": 0.58,
            },
            content_features=_feature(index),
            combined_features=np.pad(_feature(index), (0, 475)),
        )
        for index in range(6)
    ]
    images.extend(
        [
            create_analyzed_image(
                path="/tmp/other0052.jpg",
                raw_metrics_dict={
                    "brightness": 222.0,
                    "contrast": 5.0,
                    "edge_density": 0.025,
                    "action_intensity": 1.0,
                    "luminance_entropy": 0.66,
                    "luminance_range": 9.0,
                    "near_white_ratio": 0.38,
                    "dominant_tone_ratio": 0.88,
                },
                content_features=_feature(10),
                combined_features=np.pad(_feature(10), (0, 475)),
            ),
            create_analyzed_image(
                path="/tmp/event0005.jpg",
                raw_metrics_dict={
                    "brightness": 52.0,
                    "contrast": 5.5,
                    "edge_density": 0.03,
                    "action_intensity": 1.0,
                    "luminance_entropy": 0.76,
                    "luminance_range": 11.0,
                    "near_black_ratio": 0.22,
                    "near_white_ratio": 0.02,
                    "dominant_tone_ratio": 0.82,
                },
                content_features=_feature(11),
                combined_features=np.pad(_feature(11), (0, 475)),
            ),
        ]
    )

    result = ContentFilter(WholeInputProfiler()).filter(images)

    assert "/tmp/other0052.jpg" not in {image.path for image in result.kept_images}
    assert "/tmp/event0005.jpg" not in {image.path for image in result.kept_images}
    assert (
        result.content_filter_breakdown["whiteout"]
        + result.content_filter_breakdown["fade_transition"]
        >= 2
    )


def test_content_filter_rejects_temporal_transition_only_for_middle_frame() -> None:
    """通常 -> 暗転途中 -> 通常 の中央だけが temporal 判定で落ちること.

    Given:
        - 通常フレーム、遷移途中フレーム、通常フレームの順序で画像群がある
        - 前後フレームは互いに似ている
    When:
        - ContentFilterでフィルタリングする
    Then:
        - 中央の遷移途中フレームだけがtemporal_transitionとして除外されること
    """
    # Arrange
    prev_frame = create_analyzed_image(
        path="/tmp/frame_001.jpg",
        raw_metrics_dict={
            "contrast": 18.0,
            "edge_density": 0.18,
            "action_intensity": 10.0,
            "luminance_entropy": 1.3,
            "luminance_range": 38.0,
            "near_black_ratio": 0.15,
            "dominant_tone_ratio": 0.62,
        },
        content_features=_feature(10),
        combined_features=np.pad(_feature(10), (0, 475)),
    )
    mid_transition = create_analyzed_image(
        path="/tmp/frame_002.jpg",
        raw_metrics_dict={
            "brightness": 38.0,
            "contrast": 9.5,
            "edge_density": 0.08,
            "action_intensity": 4.5,
            "luminance_entropy": 0.8,
            "luminance_range": 28.0,
            "near_black_ratio": 0.18,
            "dominant_tone_ratio": 0.62,
        },
        content_features=_feature(11),
        combined_features=np.pad(_feature(11), (0, 475)),
    )
    next_frame = create_analyzed_image(
        path="/tmp/frame_003.jpg",
        raw_metrics_dict={
            "contrast": 17.0,
            "edge_density": 0.17,
            "action_intensity": 9.0,
            "luminance_entropy": 1.25,
            "luminance_range": 37.0,
            "near_black_ratio": 0.16,
            "dominant_tone_ratio": 0.63,
        },
        content_features=_feature(10, 12, 0.01),
        combined_features=np.pad(_feature(10, 12, 0.01), (0, 475)),
    )

    # Act
    result = ContentFilter(WholeInputProfiler()).filter(
        [prev_frame, mid_transition, next_frame]
    )

    # Assert
    assert {image.path for image in result.kept_images} == {
        "/tmp/frame_001.jpg",
        "/tmp/frame_003.jpg",
    }
    assert (
        result.content_filter_breakdown["fade_transition"]
        + result.content_filter_breakdown["temporal_transition"]
        == 1
    )


def test_temporal_transition_not_triggered_for_dissimilar_neighbors() -> None:
    """前後フレームが似ていなければ temporal 判定は発火しないこと.

    Given:
        - 前後フレームが互いに似ていない画像群がある
        - 中央のフレームは遷移途中のような特徴を持つ
    When:
        - ContentFilterでフィルタリングする
    Then:
        - すべてのフレームが保持されること
        - temporal_transitionによる除外が発生しないこと
    """
    # Arrange
    mixed_prev = create_analyzed_image(
        path="/tmp/mixed_prev.jpg",
        raw_metrics_dict={
            "contrast": 18.0,
            "edge_density": 0.18,
            "action_intensity": 10.0,
            "luminance_entropy": 1.3,
            "luminance_range": 38.0,
            "near_black_ratio": 0.15,
            "dominant_tone_ratio": 0.62,
        },
        content_features=_feature(20),
        combined_features=np.pad(_feature(20), (0, 475)),
    )
    mixed_current = create_analyzed_image(
        path="/tmp/mixed_current.jpg",
        raw_metrics_dict={
            "brightness": 68.0,
            "contrast": 12.0,
            "edge_density": 0.10,
            "action_intensity": 5.5,
            "luminance_entropy": 0.95,
            "luminance_range": 32.0,
            "near_black_ratio": 0.12,
            "dominant_tone_ratio": 0.58,
        },
        content_features=_feature(21),
        combined_features=np.pad(_feature(21), (0, 475)),
    )
    mixed_next = create_analyzed_image(
        path="/tmp/mixed_next.jpg",
        raw_metrics_dict={
            "contrast": 17.0,
            "edge_density": 0.17,
            "action_intensity": 9.0,
            "luminance_entropy": 1.25,
            "luminance_range": 37.0,
            "near_black_ratio": 0.16,
            "dominant_tone_ratio": 0.63,
        },
        content_features=_feature(30),
        combined_features=np.pad(_feature(30), (0, 475)),
    )

    # Act
    result = ContentFilter(WholeInputProfiler()).filter(
        [mixed_prev, mixed_current, mixed_next]
    )

    # Assert
    assert {image.path for image in result.kept_images} == {
        "/tmp/mixed_prev.jpg",
        "/tmp/mixed_current.jpg",
        "/tmp/mixed_next.jpg",
    }
    assert result.content_filter_breakdown["temporal_transition"] == 0


def test_load_image_files_returns_natural_order() -> None:
    """入力ファイルは自然順で返ること.

    Given:
        - 数字を含むファイル名が混在したディレクトリがある
        - frame1.jpg, frame10.jpg, frame2.jpg の順で作成される
    When:
        - load_image_filesでファイル一覧を取得する
    Then:
        - 自然順（frame1, frame2, frame10）で返されること
    """
    # Arrange
    with tempfile.TemporaryDirectory() as temp_dir:
        Path(temp_dir, "frame10.jpg").touch()
        Path(temp_dir, "frame2.jpg").touch()
        Path(temp_dir, "frame1.jpg").touch()

        files = GameScreenPicker.load_image_files(temp_dir, recursive=False)

    assert [path.name for path in files] == [
        "frame1.jpg",
        "frame2.jpg",
        "frame10.jpg",
    ]
