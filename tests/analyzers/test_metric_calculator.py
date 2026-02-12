"""MetricCalculatorの単体テスト.

このテストモジュールは以下のベストプラクティスに従っています：
1.「How」（実装詳細）ではなく「What」（観察可能な挙動）をテスト
2. CLIPモデルを戦略的にモック化（700MB、10-30秒のロード時間）
3. OpenCV操作、NumPy計算はモック化しない
4. 明確なコメント付きのAAAパターン（Arrange, Act, Assert）を使用
5. 高速実行（約2-5秒） - 重いモデルロードなし
"""

from typing import Any

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
    "input_type,semantic_input",
    [
        ("pil_image", None),  # 後でfixtureから取得
        ("features", np.ones(512) / np.sqrt(512.0)),
    ],
)
def test_calculate_semantic_score_returns_value_in_expected_range(
    metric_calculator: MetricCalculator,
    sample_image_path: str,
    input_type: str,
    semantic_input: Any,
) -> None:
    """セマンティックスコアが期待される範囲で返されること.

    Given:
        - メトリクス計算器がある
        - {input_type}からの入力がある
    When:
        - セマンティックスコアが計算される
    Then:
        - スコアがコサイン類似度の範囲（[-1, 1]）にあること
    """
    # Arrange
    if input_type == "pil_image":
        with Image.open(sample_image_path) as img:
            semantic_input = img.convert("RGB")

    # Act
    semantic_score = (
        metric_calculator.calculate_semantic_score(semantic_input)
        if input_type == "pil_image"
        else metric_calculator.calculate_semantic_score_from_features(semantic_input)
    )

    # Assert
    assert isinstance(semantic_score, float)
    assert not np.isnan(semantic_score)
    # 浮動小数点の丸め誤差を許容して境界チェック
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


def test_calculate_all_metrics_returns_complete_results(
    metric_calculator: MetricCalculator, sample_image_path: str
) -> None:
    """すべてのメトリクスが一括計算できること.

    Given:
        - メトリクス計算器がある
        - テスト画像がある
    When:
        - すべてのメトリクスが一括計算される
    Then:
        - 期待されるすべての結果が返されること
        - 各結果が正しい型であること
    """
    # Arrange
    img = cv2.imread(sample_image_path)
    clip_features = np.ones(512) / np.sqrt(512.0)

    # Act
    raw, norm, semantic, total = metric_calculator.calculate_all_metrics(
        img, clip_features
    )

    # Assert
    assert isinstance(raw, dict)
    assert len(raw) > 0
    assert isinstance(norm, dict)
    assert len(norm) > 0
    assert isinstance(semantic, float)
    assert isinstance(total, float)
    assert total >= 0.0
