"""StaticRejectClassifierの単体テスト."""

import numpy as np

from src.models.adaptive_scores import AdaptiveScores
from src.models.analyzed_image import AnalyzedImage
from src.models.content_reject_reason import ContentRejectReason
from src.services.static_reject_classifier import StaticRejectClassifier
from tests.conftest import build_whole_input_profile, create_analyzed_image


def _create_frame(
    path: str,
    raw_metrics: dict[str, float],
    content_features: list[float],
    normalized_metrics: dict[str, float] | None = None,
) -> AnalyzedImage:
    """静的リジェクト分類用の解析済み画像を作成する."""
    return create_analyzed_image(
        path=path,
        raw_metrics_dict=raw_metrics,
        normalized_metrics_dict=normalized_metrics,
        content_features=np.array(content_features, dtype=np.float32),
    )


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
    informative = _create_frame(
        "/tmp/informative.jpg",
        {
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
        [1.0, 0.0, 0.0],
        {
            "contrast": 0.8,
            "edge_density": 0.8,
            "action_intensity": 0.8,
        },
    )
    blackout = _create_frame(
        "/tmp/blackout.jpg",
        {
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
        [0.0, 1.0, 0.0],
    )
    whiteout = _create_frame(
        "/tmp/whiteout.jpg",
        {
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
        [0.0, 0.0, 1.0],
    )
    single_tone = _create_frame(
        "/tmp/single_tone.jpg",
        {
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
        [0.1, 0.0, 1.0],
    )
    fade_transition = _create_frame(
        "/tmp/fade_transition.jpg",
        {
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
        [0.2, 0.0, 1.0],
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
        "/tmp/blackout.jpg": ContentRejectReason.BLACKOUT,
        "/tmp/whiteout.jpg": ContentRejectReason.WHITEOUT,
        "/tmp/single_tone.jpg": ContentRejectReason.SINGLE_TONE,
        "/tmp/fade_transition.jpg": ContentRejectReason.FADE_TRANSITION,
    }
