"""MetricCalculatorの単体テスト.

このテストモジュールは以下のベストプラクティスに従っています：
1. 「What」（観察可能な挙動）をテストし、「How」（実装詳細）を検証しない
2. AAAパターン（Arrange, Act, Assert）を明確なコメントと共に使用
3. 関数ベースのテスト構造（クラスベースではない）
"""

import cv2
import numpy as np
import pytest
from PIL import Image

from src.analyzers.clip_model_manager import CLIPModelManager
from src.analyzers.metric_calculator import MetricCalculator
from src.constants.score_weights import ScoreWeights
from src.models.analyzer_config import AnalyzerConfig
from src.models.normalized_metrics import NormalizedMetrics
from src.models.raw_metrics import RawMetrics


@pytest.fixture
def metric_calculator() -> MetricCalculator:
    """メトリクス計算器のフィクスチャ."""
    config = AnalyzerConfig()
    weights = ScoreWeights.get_weights()
    # MPS環境でのテスト安定のためCPUデバイスを指定
    model_manager = CLIPModelManager(device="cpu")
    return MetricCalculator(config, weights, model_manager)


def test_calculate_raw_metrics_returns_expected_metrics(
    metric_calculator: MetricCalculator, sample_image_path: str
) -> None:
    """生メトリクスが正しい形式で返されること.

    Given:
        - メトリクス計算器がある
        - テスト画像がある
    When:
        - 生メトリクスが計算される
    Then:
        - 期待されるすべてのメトリクス属性が含まれていること
        - すべての値が数値であること
    """
    # Arrange
    img = cv2.imread(sample_image_path)

    # Act
    raw_metrics = metric_calculator.calculate_raw_metrics(img)

    # Assert: 属性の存在と値の型チェック
    expected_attrs = [
        "blur_score",
        "brightness",
        "contrast",
        "edge_density",
        "color_richness",
        "ui_density",
        "action_intensity",
        "visual_balance",
        "dramatic_score",
    ]
    for attr in expected_attrs:
        value = getattr(raw_metrics, attr)
        assert isinstance(value, (int, float))
        assert not np.isnan(value)


def test_calculate_semantic_score_returns_value_in_expected_range(
    metric_calculator: MetricCalculator,
    sample_image_path: str,
) -> None:
    """セマンティックスコアが期待される範囲で返されること.

    Given:
        - メトリクス計算器がある
        - テスト画像がある
    When:
        - セマンティックスコアが計算される
    Then:
        - スコアがコサイン類似度の範囲（[-1, 1]）にあること
    """
    # Arrange
    with Image.open(sample_image_path) as img:
        pil_img = img.convert("RGB")

    # Act
    semantic_score = metric_calculator.calculate_semantic_score(pil_img)

    # Assert
    assert isinstance(semantic_score, float)
    assert not np.isnan(semantic_score)
    assert -1.0 <= semantic_score <= 1.0 + 1e-5


def test_calculate_total_score_returns_non_negative_value(
    metric_calculator: MetricCalculator,
) -> None:
    """総合スコアが負ではない値で返されること.

    Given:
        - メトリクス計算器がある
        - 有効なメトリクスがある
    When:
        - 総合スコアが計算される
    Then:
        - スコアが0以上であること
    """
    # Arrange
    raw = RawMetrics(
        blur_score=500.0,
        brightness=100.0,
        contrast=50.0,
        edge_density=0.2,
        color_richness=40.0,
        ui_density=10.0,
        action_intensity=30.0,
        visual_balance=90.0,
        dramatic_score=50.0,
    )

    norm = NormalizedMetrics(
        blur_score=1.0,
        contrast=1.0,
        color_richness=1.0,
        edge_density=1.0,
        dramatic_score=0.5,
        visual_balance=0.9,
        action_intensity=1.0,
        ui_density=1.0,
    )
    semantic = 0.5

    # Act
    total_score = metric_calculator.calculate_total_score(raw, norm, semantic)

    # Assert
    assert total_score >= 0.0
    assert isinstance(total_score, float)
