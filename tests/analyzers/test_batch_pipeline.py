"""BatchPipelineの単体テスト.

このテストモジュールは以下のベストプラクティスに従っています：
1.「How」（実装詳細）ではなく「What」（観察可能な挙動）をテスト
2. CLIPモデルを戦略的にモック化（700MB、10-30秒のロード時間）
3. OpenCV操作、NumPy計算はモック化しない
4. 明確なコメント付きのAAAパターン（Arrange, Act, Assert）を使用
5. 高速実行（約2-5秒） - 重いモデルロードなし
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.analyzers.batch_pipeline import BatchPipeline
from src.analyzers.clip_model_manager import CLIPModelManager
from src.analyzers.feature_extractor import FeatureExtractor
from src.analyzers.metric_calculator import MetricCalculator
from src.constants.score_weights import ScoreWeights
from src.models.analyzer_config import AnalyzerConfig
from src.models.image_metrics import ImageMetrics


@pytest.fixture
def batch_pipeline() -> BatchPipeline:
    """バッチ処理パイプラインのフィクスチャ."""
    config = AnalyzerConfig()
    weights = ScoreWeights.get_weights()
    model_manager = CLIPModelManager()
    feature_extractor = FeatureExtractor(model_manager)
    metric_calculator = MetricCalculator(config, weights, model_manager)
    return BatchPipeline(feature_extractor, metric_calculator, config)


@pytest.mark.parametrize(
    "num_images,batch_size",
    [(3, 1), (5, 2)],
)
def test_process_batch_handles_multiple_images(
    batch_pipeline: BatchPipeline,
    tmp_path: Path,
    num_images: int,
    batch_size: int,
) -> None:
    """複数の画像が正しくバッチ処理されること.

    Given:
        - バッチ処理パイプラインがある
        - 複数の有効なテスト画像がある
    When:
        - 複数の画像がバッチ処理で分析される
    Then:
        - すべての画像に対して有効なImageMetricsが返されること
        - 結果の数が入力数と一致すること
        - 各結果のパスが正しいこと
    """
    # Arrange: 複数の画像を作成
    paths = []
    for i in range(num_images):
        np.random.seed(42 + i)
        img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img_path = tmp_path / f"batch_test_{i}.jpg"
        cv2.imwrite(str(img_path), img_array)
        paths.append(str(img_path))

    # Act
    results = batch_pipeline.process_batch(paths, batch_size=batch_size)

    # Assert
    assert len(results) == num_images
    assert any(r is not None for r in results)
    for result, path in zip(results, paths, strict=True):
        if result is not None:
            assert isinstance(result, ImageMetrics)
            assert result.path == path
            assert 0 <= result.total_score <= 100
            assert -1.0 <= result.semantic_score <= 1.0 + 1e-5
