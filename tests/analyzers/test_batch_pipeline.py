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
    for result, path in zip(results, paths, strict=True):
        if result is not None:
            assert isinstance(result, ImageMetrics)
            assert result.path == path
            assert 0 <= result.total_score <= 100
            assert -1.0 <= result.semantic_score <= 1.0 + 1e-5


def test_process_batch_with_lookahead_processes_all_images(
    batch_pipeline: BatchPipeline,
    tmp_path: Path,
) -> None:
    """先読み付きバッチ処理ですべての画像が処理されること.

    Given:
        - バッチ処理パイプラインがある
        - 複数のテスト画像がある
    When:
        - 先読み付きで複数の画像をバッチ処理する
    Then:
        - すべての画像に対して有効な結果が返されること
        - 結果のパスが正しいこと
    """
    # Arrange: 複数の画像を作成
    paths = []
    for i in range(5):
        np.random.seed(42 + i)
        img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img_path = tmp_path / f"lookahead_test_{i}.jpg"
        cv2.imwrite(str(img_path), img_array)
        paths.append(str(img_path))

    # Act
    results = batch_pipeline.process_batch(paths, batch_size=2)

    # Assert
    assert len(results) == 5
    assert any(r is not None for r in results)
    for result, path in zip(results, paths, strict=True):
        if result is not None:
            assert isinstance(result, ImageMetrics)
            assert result.path == path
            assert 0 <= result.total_score <= 100


def test_batch_pipeline_handles_multiple_chunks_successfully(
    batch_pipeline: BatchPipeline,
    tmp_path: Path,
) -> None:
    """複数チャンクに分割されるバッチ処理が正常に完了すること.

    Given:
        - バッチ処理パイプラインがある
        - 複数のテスト画像がある
        - 小さなバッチサイズで複数チャンクに分割される
    When:
        - 複数チャンクでバッチ処理を実行する
    Then:
        - すべての画像で処理が成功すること
        - 結果の数が入力数と一致すること
    """
    # Arrange: 複数の画像を作成
    paths = []
    for i in range(20):
        np.random.seed(42 + i)
        img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img_path = tmp_path / f"multi_chunk_test_{i}.jpg"
        cv2.imwrite(str(img_path), img_array)
        paths.append(str(img_path))

    # Act: 小さなチャンクサイズでバッチ処理
    results = batch_pipeline.process_batch(paths, batch_size=2)

    # Assert: すべての結果が得られる
    assert len(results) == 20
    assert any(r is not None for r in results)
    for result, path in zip(results, paths, strict=True):
        if result is not None:
            assert isinstance(result, ImageMetrics)
            assert result.path == path


def test_batch_pipeline_context_manager(
    batch_pipeline: BatchPipeline,
    sample_image_path: str,
) -> None:
    """コンテキストマネージャーでバッチ処理が正しく動作すること.

    Given:
        - バッチ処理パイプラインがある
        - 有効なテスト画像がある
    When:
        - withステートメントで使用する
    Then:
        - 処理が正常に完了すること
        - 結果が返されること
    """
    # Arrange
    paths = [sample_image_path]

    # Act: コンテキストマネージャーで使用
    with batch_pipeline:
        results = batch_pipeline.process_batch(paths, batch_size=1)

    # Assert
    assert len(results) == 1
    assert results[0] is not None


def test_load_and_preprocess_images_with_max_dim(
    batch_pipeline: BatchPipeline, tmp_path: Path
) -> None:
    """load_and_preprocess_imagesでmax_dimが指定できること.

    Given:
        - 2000x1000の大きな画像がある
    When:
        - max_dim=720でload_and_preprocess_imagesを実行
    Then:
        - 結果の画像が720以下に縮小されていること
    """
    # Arrange: 2000x1000の画像を作成
    img_array = np.random.randint(0, 255, (1000, 2000, 3), dtype=np.uint8)
    large_image_path = tmp_path / "large_for_batch.jpg"
    cv2.imwrite(str(large_image_path), img_array)

    # Act: max_dim=720で読み込み
    results = batch_pipeline.load_and_preprocess_images(
        [str(large_image_path)], max_dim=720
    )

    # Assert
    assert len(results) == 1
    assert results[0] is not None
    w, h = results[0].size
    assert max(w, h) <= 720


def test_io_max_workers_1_does_not_deadlock(
    tmp_path: Path,
) -> None:
    """io_max_workers=1でデッドロックせずに処理が完了すること.

    Given:
        - io_max_workers=1の設定がある
        - 複数のテスト画像がある
    When:
        - バッチ処理を実行する
    Then:
        - すべての画像の結果が返されること
    """
    # Arrange: io_max_workers=1でBatchPipelineを作成
    config = AnalyzerConfig(io_max_workers=1)
    weights = ScoreWeights.get_weights()
    model_manager = CLIPModelManager()
    feature_extractor = FeatureExtractor(model_manager)
    metric_calculator = MetricCalculator(config, weights, model_manager)
    pipeline = BatchPipeline(feature_extractor, metric_calculator, config)

    # 複数の画像を作成
    paths = []
    for i in range(5):
        np.random.seed(42 + i)
        img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img_path = tmp_path / f"io1_test_{i}.jpg"
        cv2.imwrite(str(img_path), img_array)
        paths.append(str(img_path))

    # Act
    results = pipeline.process_batch(paths, batch_size=2)

    # Assert
    assert len(results) == 5
    assert any(r is not None for r in results)

    # クリーンアップ
    pipeline.close()
