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
from src.analyzers.metric_normalizer import MetricNormalizer
from src.constants.genre_weights import GenreWeights
from src.models.analyzer_config import AnalyzerConfig


@pytest.fixture
def metric_calculator() -> MetricCalculator:
    """メトリクス計算器のフィクスチャ."""
    config = AnalyzerConfig()
    weights = GenreWeights.get_weights("mixed")
    model_manager = CLIPModelManager()
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
        - 期待されるすべてのメトリクスキーが含まれていること
        - すべての値が数値であること
    """
    # Arrange
    img = cv2.imread(sample_image_path)

    # Act
    raw_metrics = metric_calculator.calculate_raw_metrics(img)

    # Assert
    expected_keys = {
        "blur_score",
        "brightness",
        "contrast",
        "edge_density",
        "color_richness",
        "ui_density",
        "action_intensity",
        "visual_balance",
        "dramatic_score",
    }
    assert set(raw_metrics.keys()) == expected_keys
    for value in raw_metrics.values():
        assert isinstance(value, (int, float))
        assert not np.isnan(value)


@pytest.mark.parametrize(
    "use_image_input",
    [True, False],
)
def test_calculate_semantic_score_returns_value_in_expected_range(
    metric_calculator: MetricCalculator,
    sample_image_path: str,
    use_image_input: bool,
) -> None:
    """セマンティックスコアが期待される範囲で返されること.

    Given:
        - メトリクス計算器がある
        - PIL画像または正規化された特徴ベクトルがある
    When:
        - セマンティックスコアが計算される
    Then:
        - スコアがコサイン類似度の範囲（[-1, 1]）にあること
    """
    # Arrange
    if use_image_input:
        with Image.open(sample_image_path) as img:
            pil_img = img.convert("RGB")
    else:
        features = np.ones(512) / np.sqrt(512.0)

    # Act
    if use_image_input:
        semantic_score = metric_calculator.calculate_semantic_score(pil_img)
    else:
        semantic_score = metric_calculator.calculate_semantic_score_from_features(
            features
        )

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
    raw = {
        "blur_score": 500.0,
        "brightness": 100.0,
        "contrast": 50.0,
        "edge_density": 0.2,
        "color_richness": 40.0,
        "ui_density": 10.0,
        "action_intensity": 30.0,
        "visual_balance": 90.0,
        "dramatic_score": 50.0,
    }

    norm = MetricNormalizer.normalize_all(raw)
    semantic = 0.5

    # Act
    total_score = metric_calculator.calculate_total_score(raw, norm, semantic)

    # Assert
    assert total_score >= 0.0
    assert isinstance(total_score, float)
