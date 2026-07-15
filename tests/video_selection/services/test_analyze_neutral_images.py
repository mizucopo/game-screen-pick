"""Neutral Image Analysisのtest。"""

from fractions import Fraction

import cv2
import numpy as np
import pytest

from src.video_selection.models.content_reject_reason import ContentRejectReason
from src.video_selection.models.decoded_video_frame import DecodedVideoFrame
from src.video_selection.services.analyze_neutral_images import analyze_neutral_images


def _frame(source_pts: int, rgb: np.ndarray) -> DecodedVideoFrame:
    height, width = rgb.shape[:2]
    return DecodedVideoFrame(
        stream_index=0,
        pts=source_pts,
        duration_ts=1,
        time_base=Fraction(1, 10),
        width=width,
        height=height,
        pixel_format="rgb24",
        pixels=rgb.astype(np.uint8).tobytes(),
    )


def _checkerboard() -> np.ndarray:
    rows, columns = np.indices((48, 64))
    values = ((rows // 4 + columns // 4) % 2 * 180 + 35).astype(np.uint8)
    return np.stack((values, np.roll(values, 5, axis=1), 255 - values), axis=2)


@pytest.mark.parametrize(
    ("rgb", "expected_reason"),
    [
        pytest.param(
            np.zeros((48, 64, 3), dtype=np.uint8),
            ContentRejectReason.BLACKOUT,
            id="blackout",
        ),
        pytest.param(
            np.full((48, 64, 3), 255, dtype=np.uint8),
            ContentRejectReason.WHITEOUT,
            id="whiteout",
        ),
        pytest.param(
            np.full((48, 64, 3), (35, 90, 145), dtype=np.uint8),
            ContentRejectReason.SINGLE_TONE,
            id="single-tone",
        ),
    ],
)
def test_absolute_invalid_frame_has_stable_reject_reason(
    rgb: np.ndarray,
    expected_reason: ContentRejectReason,
) -> None:
    """明確な無効frameがstable Content Reject Reasonで除外されること。

    Arrange:
        - 黒、白、または単色のRGB frameが用意される
    Act:
        - model-free Neutral Image Analysisが実行される
    Assert:
        - 対応するstable reasonが返されること
    """
    # Arrange
    frame = _frame(0, rgb)

    # Act
    analysis = analyze_neutral_images((frame,))[0]

    # Assert
    assert analysis.reject_reason is expected_reason
    assert not analysis.eligible


def test_smooth_blurred_frame_is_rejected_without_model_inference() -> None:
    """低周波だけのぼけframeがversioned閾値で除外されること。

    Arrange:
        - 色と輝度範囲はあるがedgeを持たない滑らかなframeが用意される
    Act:
        - Neutral Image Analysisが実行される
    Assert:
        - blurとして除外されること
    """
    # Arrange
    gradient = np.tile(np.linspace(20, 220, 64, dtype=np.uint8), (48, 1))
    smooth = cv2.GaussianBlur(
        np.stack((gradient, np.flipud(gradient), gradient), axis=2),
        (15, 15),
        0,
    )

    # Act
    analysis = analyze_neutral_images((_frame(0, smooth),))[0]

    # Assert
    assert analysis.reject_reason is ContentRejectReason.BLUR


def test_visual_feature_is_l2_normalized_and_temporal_transition_is_rejected() -> None:
    """model-free視覚特徴と前後関係から遷移frameが解析されること。

    Arrange:
        - 同じ高情報frameの間に暗いoverlay frameが置かれる
    Act:
        - 動画内分布を使うNeutral Image Analysisが実行される
    Assert:
        - 有効frameのHSV・輝度・edge特徴がL2正規化されること
        - 中央frameだけがtemporal transitionとして除外されること
    """
    # Arrange
    detailed = _checkerboard()
    dark_overlay = (detailed.astype(np.float32) * 0.22 + 12).astype(np.uint8)
    frames = (
        _frame(0, detailed),
        _frame(1, dark_overlay),
        _frame(2, detailed),
    )

    # Act
    analyses = analyze_neutral_images(frames)

    # Assert
    assert analyses[0].reject_reason is None
    assert analyses[2].reject_reason is None
    assert np.linalg.norm(analyses[0].visual_feature) == pytest.approx(1.0)
    assert analyses[1].reject_reason is ContentRejectReason.TEMPORAL_TRANSITION


def test_temporal_transition_requires_exact_native_frame_adjacency() -> None:
    """離れたsampleがtemporal transitionの前後frameにされないこと。

    Arrange:
        - 同じ高情報frameの間に暗いoverlayが離れたPTSで用意される
    Act:
        - Neutral Image Analysisが実行される
    Assert:
        - 中央sampleがtemporal transitionとして除外されないこと
    """
    # Arrange
    detailed = _checkerboard()
    dark_overlay = (detailed.astype(np.float32) * 0.22 + 12).astype(np.uint8)
    frames = (
        _frame(0, detailed),
        _frame(10, dark_overlay),
        _frame(20, detailed),
    )

    # Act
    analyses = analyze_neutral_images(frames)

    # Assert
    assert analyses[1].reject_reason is not ContentRejectReason.TEMPORAL_TRANSITION
