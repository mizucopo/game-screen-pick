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


def test_large_white_effects_are_rejected_without_pure_white_frame() -> None:
    """主対象を覆う白い発光と白いveilがwhiteoutとして除外されること。

    Arrange:
        - 中央を白い発光が覆うframeと、画面の大半が淡い白になるframeが用意される
    Act:
        - Neutral Image Analysisが実行される
    Assert:
        - 完全な白一色でなくても両方がwhiteoutとして除外されること
    """
    # Arrange
    central_flash = cv2.resize(
        _checkerboard(),
        (960, 540),
        interpolation=cv2.INTER_NEAREST,
    )
    central_flash[90:450, 180:780] = 250
    soft_white_veil = np.full((540, 960, 3), 232, dtype=np.uint8)
    soft_white_veil[:, 900:] = cv2.resize(
        _checkerboard(),
        (60, 540),
        interpolation=cv2.INTER_NEAREST,
    )

    # Act
    analyses = analyze_neutral_images(
        (
            _frame(0, central_flash),
            _frame(10, soft_white_veil),
        )
    )

    # Assert
    assert [item.reject_reason for item in analyses] == [
        ContentRejectReason.WHITEOUT,
        ContentRejectReason.WHITEOUT,
    ]


def test_bright_menu_with_distinct_structure_remains_eligible() -> None:
    """明るくても構造が判別できるmenu frameが除外されないこと。

    Arrange:
        - 淡い背景へ濃い罫線と区画を持つmenu風frameが用意される
    Act:
        - Neutral Image Analysisが実行される
    Assert:
        - 明るさだけを理由にwhiteoutとして除外されないこと
    """
    # Arrange
    menu = np.full((540, 960, 3), 250, dtype=np.uint8)
    for row in range(60, 500, 55):
        cv2.line(menu, (80, row), (880, row), (45, 55, 65), 4)
    cv2.rectangle(menu, (610, 80), (880, 430), (35, 70, 110), 8)

    # Act
    analysis = analyze_neutral_images((_frame(0, menu),))[0]

    # Assert
    assert analysis.reject_reason is None
    assert analysis.eligible


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


def test_expanding_soft_bright_sweep_is_rejected_as_temporal_transition() -> None:
    """短時間に拡大する淡い白の移動演出が遷移frameとして除外されること。

    Arrange:
        - 情報のある背景を淡い白領域が連続3frameで急速に覆う場面が用意される
    Act:
        - 前後関係を使うNeutral Image Analysisが実行される
    Assert:
        - 純白でなくても移動演出中の3frameがすべて除外されること
    """
    # Arrange
    detailed = cv2.resize(
        _checkerboard(),
        (960, 540),
        interpolation=cv2.INTER_NEAREST,
    )
    sweep_frames = []
    for covered_width in (100, 660, 900):
        swept = detailed.copy()
        swept[:, :covered_width] = 225
        sweep_frames.append(swept)

    # Act
    analyses = analyze_neutral_images(
        tuple(_frame(index, frame) for index, frame in enumerate(sweep_frames))
    )

    # Assert
    assert [item.reject_reason for item in analyses] == [
        ContentRejectReason.TEMPORAL_TRANSITION,
        ContentRejectReason.TEMPORAL_TRANSITION,
        ContentRejectReason.TEMPORAL_TRANSITION,
    ]


def test_static_bright_interface_is_not_a_soft_bright_sweep() -> None:
    """静止した明るいinterfaceが移動演出として誤除外されないこと。

    Arrange:
        - 広い淡色領域と濃い罫線を持つ同一menu frameが連続して用意される
    Act:
        - 前後関係を使うNeutral Image Analysisが実行される
    Assert:
        - 明るい領域が変化しないframeはすべて有効なままであること
    """
    # Arrange
    menu = np.full((540, 960, 3), 225, dtype=np.uint8)
    for row in range(50, 500, 50):
        cv2.line(menu, (60, row), (900, row), (40, 55, 70), 5)
    cv2.rectangle(menu, (600, 70), (900, 450), (30, 65, 100), 10)
    frames = tuple(_frame(index, menu) for index in range(3))

    # Act
    analyses = analyze_neutral_images(frames)

    # Assert
    assert all(item.reject_reason is None for item in analyses)
