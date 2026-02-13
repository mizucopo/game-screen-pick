"""MetricCalculatorの単体テスト.

このテストモジュールは以下のベストプラクティスに従っています：
1. 「What」（観察可能な挙動）をテストし、「How」（実装詳細）を検証しない
2. AAAパターン（Arrange, Act, Assert）を明確なコメントと共に使用
3. 関数ベースのテスト構造（クラスベースではない）
"""

import cv2
import numpy as np
import pytest

from src.analyzers.clip_model_manager import CLIPModelManager
from src.analyzers.metric_calculator import MetricCalculator
from src.constants.score_weights import ScoreWeights
from src.models.analyzer_config import AnalyzerConfig


@pytest.fixture
def metric_calculator() -> MetricCalculator:
    """メトリクス計算器のフィクスチャ."""
    config = AnalyzerConfig()
    weights = ScoreWeights.get_weights()
    # MPS環境でのテスト安定のためCPUデバイスを指定
    model_manager = CLIPModelManager(device="cpu")
    return MetricCalculator(config, weights, model_manager)


def test_calculate_metrics_returns_valid_values(
    metric_calculator: MetricCalculator, sample_image_path: str
) -> None:
    """画像から有効なメトリクスが計算されること.

    Given:
        - メトリクス計算器がある
        - テスト画像がある
    When:
        - メトリクスが計算される
    Then:
        - 有効なメトリクスインスタンスが返されること
        - すべての数値が有効な範囲にあること
    """
    # Arrange
    img = cv2.imread(sample_image_path)

    # Act
    raw_metrics = metric_calculator.calculate_raw_metrics(img)

    # Assert: 有効な値範囲を確認
    assert raw_metrics.blur_score > 0
    assert 0 <= raw_metrics.brightness <= 255
    assert not np.isnan(raw_metrics.edge_density)
    assert not np.isnan(raw_metrics.action_intensity)


def test_semantic_score_in_cosine_similarity_range(
    metric_calculator: MetricCalculator,
    sample_image_path: str,
) -> None:
    """セマンティックスコアがコサイン類似度の範囲内にあること.

    Given:
        - メトリクス計算器がある
        - テスト画像がある
    When:
        - セマンティックスコアが計算される
    Then:
        - スコアがコサイン類似度の範囲（[-1, 1]）にあること
    """
    # Arrange
    from PIL import Image

    with Image.open(sample_image_path) as img:
        pil_img = img.convert("RGB")

    # Act
    semantic_score = metric_calculator.calculate_semantic_score(pil_img)

    # Assert
    assert not np.isnan(semantic_score)
    assert -1.0 <= semantic_score <= 1.0 + 1e-5
