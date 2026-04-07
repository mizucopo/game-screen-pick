"""MetricCalculatorの単体テスト.

このテストモジュールでは、現行実装の公開挙動として
生メトリクス計算、quality score 計算、新しい輝度分布メトリクスを検証する。
実装詳細ではなく、観測可能な戻り値だけを見る方針を取る。
"""

import cv2
import numpy as np
import pytest

from src.analyzers.metric_calculator import MetricCalculator
from src.models.analyzer_config import AnalyzerConfig


@pytest.fixture()
def metric_calculator() -> MetricCalculator:
    """メトリクス計算器のfixture.

    scene mix 対応後の `MetricCalculator` を
    デフォルト設定で使い回すための軽量fixture。
    """
    return MetricCalculator(AnalyzerConfig())


def test_calculate_metrics_returns_valid_values(
    metric_calculator: MetricCalculator, sample_image_path: str
) -> None:
    """画像から有効なメトリクスが計算されること.

    Arrange:
        - メトリクス計算器がある
        - 読み込み可能なテスト画像がある
    Act:
        - 生メトリクスを計算する
    Assert:
        - 主要メトリクスが期待範囲の値として返ること
    """
    # Arrange
    img = cv2.imread(sample_image_path)
    assert img is not None, f"テスト画像が読み込めません: {sample_image_path}"

    # Act
    raw_metrics = metric_calculator.calculate_raw_metrics(img)

    # Assert
    assert raw_metrics.blur_score > 0
    assert 0 <= raw_metrics.brightness <= 255
    assert not np.isnan(raw_metrics.edge_density)
    assert not np.isnan(raw_metrics.action_intensity)


def test_quality_score_uses_weights(metric_calculator: MetricCalculator) -> None:
    """quality_scoreが指定重みで計算されること.

    Arrange:
        - 正規化メトリクスを計算できる画像がある
        - 一部の項目だけに重みを持つ quality weight がある
    Act:
        - quality score を計算する
    Assert:
        - 0.0 以上 1.0 以下のスコアとして返ること
    """
    # Arrange
    _, norm = metric_calculator.calculate_all_metrics(
        np.full((32, 32, 3), 180, dtype=np.uint8)
    )
    weights = {
        "blur_score": 0.5,
        "contrast": 0.5,
        "color_richness": 0.0,
        "edge_density": 0.0,
        "dramatic_score": 0.0,
        "visual_balance": 0.0,
        "action_intensity": 0.0,
        "ui_density": 0.0,
    }

    # Act
    score = metric_calculator.calculate_quality_score(norm, weights)

    # Assert
    assert 0.0 <= score <= 1.0


def test_calculate_metrics_distinguishes_flat_and_textured_frames(
    metric_calculator: MetricCalculator,
) -> None:
    """black/white/single-tone と暗い高情報量画像が分離されること.

    Arrange:
        - 真っ黒、真っ白、単色、暗いが情報量のある画像がある
    Act:
        - 生メトリクスを計算する
    Assert:
        - 平坦な画像群では分布メトリクスが低くなり
        - 暗いが情報量のある画像では entropy/range/edge が高くなること
    """
    # Arrange
    black_image = np.zeros((64, 64, 3), dtype=np.uint8)
    white_image = np.full((64, 64, 3), 255, dtype=np.uint8)
    single_tone_image = np.full((64, 64, 3), (0, 220, 255), dtype=np.uint8)
    dark_textured_image = np.zeros((64, 64, 3), dtype=np.uint8) + 20
    for offset in range(0, 64, 8):
        cv2.line(dark_textured_image, (0, offset), (63, offset), (40, 40, 40), 1)
        cv2.line(dark_textured_image, (offset, 0), (63, 63 - offset), (55, 55, 55), 1)

    # Act
    black_metrics = metric_calculator.calculate_raw_metrics(black_image)
    white_metrics = metric_calculator.calculate_raw_metrics(white_image)
    single_tone_metrics = metric_calculator.calculate_raw_metrics(single_tone_image)
    dark_textured_metrics = metric_calculator.calculate_raw_metrics(dark_textured_image)

    # Assert
    assert black_metrics.near_black_ratio == pytest.approx(1.0)
    assert white_metrics.near_white_ratio == pytest.approx(1.0)
    assert single_tone_metrics.dominant_tone_ratio == pytest.approx(1.0)
    assert black_metrics.luminance_entropy == pytest.approx(0.0)
    assert white_metrics.luminance_entropy == pytest.approx(0.0)
    assert single_tone_metrics.luminance_range == pytest.approx(0.0)
    assert (
        dark_textured_metrics.luminance_entropy > single_tone_metrics.luminance_entropy
    )
    assert dark_textured_metrics.luminance_range > single_tone_metrics.luminance_range
    assert dark_textured_metrics.edge_density > single_tone_metrics.edge_density
    assert dark_textured_metrics.action_intensity > black_metrics.action_intensity
