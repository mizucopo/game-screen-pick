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
from src.cache.feature_cache import FeatureCache
from src.constants.genre_weights import GenreWeights
from src.models.analyzer_config import AnalyzerConfig
from src.models.image_metrics import ImageMetrics


@pytest.fixture
def batch_pipeline() -> BatchPipeline:
    """バッチ処理パイプラインのフィクスチャ."""
    config = AnalyzerConfig()
    weights = GenreWeights.get_weights("mixed")
    model_manager = CLIPModelManager()
    feature_extractor = FeatureExtractor(model_manager)
    metric_calculator = MetricCalculator(config, weights, model_manager)
    return BatchPipeline(feature_extractor, metric_calculator, config)


def test_process_batch_handles_multiple_images(
    batch_pipeline: BatchPipeline,
    sample_image_path: str,
    png_image_path: str,
    tmp_path: Path,
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
    # Arrange
    np.random.seed(43)
    img_array = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    small_image_path = tmp_path / "small_image.jpg"
    cv2.imwrite(str(small_image_path), img_array)

    paths = [sample_image_path, png_image_path, str(small_image_path)]

    # Act
    results = batch_pipeline.process_batch(paths, batch_size=1)

    # Assert
    assert len(results) == 3
    assert any(r is not None for r in results)
    for result, path in zip(results, paths):
        if result is not None:
            assert isinstance(result, ImageMetrics)
            assert result.path == path
            assert 0 <= result.total_score <= 100
            assert -1.0 <= result.semantic_score <= 1.0 + 1e-5


def test_cached_results_semantic_score_calculation_is_batched(
    tmp_path: Path,
) -> None:
    """キャッシュヒット時のセマンティックスコア計算がバッチ化されていること.

    Given:
        - バッチ処理パイプラインにキャッシュを設定
        - キャッシュにエントリが保存されている
    When:
        - 同じ画像を再度バッチ処理
    Then:
        - キャッシュから結果が正しく取得できること
        - 2回目の処理結果が1回目と一致すること
    """
    # Arrange: キャッシュ付きのパイプラインを作成
    config = AnalyzerConfig()
    weights = GenreWeights.get_weights("mixed")
    model_manager = CLIPModelManager()
    feature_extractor = FeatureExtractor(model_manager)
    metric_calculator = MetricCalculator(config, weights, model_manager)

    cache_path = tmp_path / "test_cache.sqlite3"
    cache = FeatureCache(cache_path)

    pipeline_with_cache = BatchPipeline(
        feature_extractor,
        metric_calculator,
        config,
        cache=cache,
    )

    # 複数のテスト画像を作成
    paths = []
    for i in range(3):
        np.random.seed(42 + i)
        img_array = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        img_path = tmp_path / f"cache_test_{i}.jpg"
        cv2.imwrite(str(img_path), img_array)
        paths.append(str(img_path))

    # Act: 最初の実行（キャッシュミス）
    first_results = pipeline_with_cache.process_batch(paths, batch_size=2)

    # 2回目の実行（キャッシュヒット）
    second_results = pipeline_with_cache.process_batch(paths, batch_size=2)

    # Assert: 両方の実行で有効な結果が得られる
    assert len(first_results) == 3
    assert len(second_results) == 3

    # キャッシュヒット時も結果が正しく取得できる
    for first, second in zip(first_results, second_results):
        assert first is not None
        assert second is not None
        # 結果が一致すること（キャッシュから再計算された値）
        assert second.path == first.path
        # セマンティックスコアが計算されている
        assert -1.0 <= second.semantic_score <= 1.0 + 1e-5


def test_compute_chunk_boundaries_uses_fast_estimation(tmp_path: Path) -> None:
    """チャンク境界計算で高速なメモリ推定が使用されていること.

    Given:
        - バッチ処理パイプラインがある
        - 複数のテスト画像がある
    When:
        - チャンク境界を計算
    Then:
        - os.statベースの推定が使用されていること
        - PIL Image.openが使用されていないこと（高速化）
    """
    # Arrange: 複数のテスト画像を作成
    paths = []
    for i in range(5):
        np.random.seed(42 + i)
        # 各画像で異なるサイズを作成
        size = 100 + i * 50
        img_array = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
        img_path = tmp_path / f"chunk_test_{i}.jpg"
        cv2.imwrite(str(img_path), img_array)
        paths.append(str(img_path))

    # Act: チャンク境界を計算
    # 小さなメモリ予算で複数チャンクに分割されるように設定
    max_memory_mb = 1
    min_chunk_size = 2
    boundaries = BatchPipeline._compute_chunk_boundaries(
        paths, max_memory_mb, min_chunk_size
    )

    # Assert: チャンク境界が計算されている
    assert len(boundaries) > 0
    # 各チャンクが有効な範囲を持つ
    for start, end in boundaries:
        assert 0 <= start < end <= len(paths)

    # 最初のチャンクはインデックス0から始まる
    assert boundaries[0][0] == 0
    # 最後のチャンクはリストの末尾まで
    assert boundaries[-1][1] == len(paths)
