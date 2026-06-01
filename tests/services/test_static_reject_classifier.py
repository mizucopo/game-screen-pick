"""StaticRejectClassifierの単体テスト."""

import numpy as np

from src.models.adaptive_scores import AdaptiveScores
from src.services.static_reject_classifier import StaticRejectClassifier
from tests.conftest import build_whole_input_profile, create_analyzed_image


def test_static_reject_classifier_classifies_low_information_frames() -> None:
    """低情報量フレームのhard reject理由が分類されること.

    Arrange:
        - 通常フレーム、ブラックアウト、ホワイトアウト、単色、フェード遷移がある
        - 入力全体の分布プロフィールが構築されている
    Act:
        - StaticRejectClassifierで各フレームが分類される
    Assert:
        - 各低情報量フレームに対応する除外理由が返されること
        - 通常フレームは除外理由なしとして扱われること
    """
    # Arrange
    informative = create_analyzed_image(
        path="/tmp/informative.jpg",
        raw_metrics_dict={
            "brightness": 120.0,
            "contrast": 55.0,
            "edge_density": 0.35,
            "action_intensity": 45.0,
            "luminance_entropy": 3.0,
            "luminance_range": 120.0,
            "near_black_ratio": 0.02,
            "near_white_ratio": 0.01,
            "dominant_tone_ratio": 0.35,
        },
        normalized_metrics_dict={
            "contrast": 0.8,
            "edge_density": 0.8,
            "action_intensity": 0.8,
        },
        content_features=np.array([1.0, 0.0, 0.0], dtype=np.float32),
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
        content_features=np.array([0.0, 1.0, 0.0], dtype=np.float32),
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
        content_features=np.array([0.0, 0.0, 1.0], dtype=np.float32),
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
        content_features=np.array([0.1, 0.0, 1.0], dtype=np.float32),
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
        content_features=np.array([0.2, 0.0, 1.0], dtype=np.float32),
    )
    profile = build_whole_input_profile(
        informative,
        blackout,
        whiteout,
        single_tone,
        fade_transition,
    )
    visible_scores = AdaptiveScores(information_score=0.9, visibility_score=0.9)
    weak_scores = AdaptiveScores(information_score=0.0, visibility_score=0.0)

    # Act
    classifier = StaticRejectClassifier()
    reasons = {
        informative.path: classifier.classify(informative, profile, visible_scores),
        blackout.path: classifier.classify(blackout, profile, weak_scores),
        whiteout.path: classifier.classify(whiteout, profile, weak_scores),
        single_tone.path: classifier.classify(single_tone, profile, weak_scores),
        fade_transition.path: classifier.classify(
            fade_transition,
            profile,
            weak_scores,
        ),
    }

    # Assert
    assert reasons == {
        "/tmp/informative.jpg": None,
        "/tmp/blackout.jpg": "blackout",
        "/tmp/whiteout.jpg": "whiteout",
        "/tmp/single_tone.jpg": "single_tone",
        "/tmp/fade_transition.jpg": "fade_transition",
    }
