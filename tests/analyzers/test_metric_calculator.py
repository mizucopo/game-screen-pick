"""MetricCalculatorの単体テスト.

このテストモジュールでは、現行実装の公開挙動として
生メトリクス計算、quality score 計算、brightness penalty を検証する。
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

    Given:
        - メトリクス計算器がある
        - 読み込み可能なテスト画像がある
    When:
        - 生メトリクスを計算する
    Then:
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

    Given:
        - 正規化メトリクスを計算できる画像がある
        - 一部の項目だけに重みを持つ quality weight がある
    When:
        - quality score を計算する
    Then:
        - 0.0 以上 1.0 以下のスコアとして返ること
    """
    # Arrange
    _, norm = metric_calculator.calculate_raw_norm_metrics(
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


def test_brightness_penalty_applies_to_dark_image(
    metric_calculator: MetricCalculator,
) -> None:
    """暗い画像にbrightness penaltyが適用されること.

    Given:
        - 明るさが極端に低い画像がある
    When:
        - brightness penalty を計算する
    Then:
        - 設定済みの penalty 値がそのまま返ること
    """
    # Arrange
    dark_image = np.zeros((32, 32, 3), dtype=np.uint8)
    raw_metrics = metric_calculator.calculate_raw_metrics(dark_image)

    # Act
    penalty = metric_calculator.calculate_brightness_penalty(raw_metrics)

    # Assert
    assert penalty == metric_calculator.config.brightness_penalty_value
