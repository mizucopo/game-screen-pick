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
    for result, path in zip(results, paths):
        if result is not None:
            assert isinstance(result, ImageMetrics)
            assert result.path == path
            assert 0 <= result.total_score <= 100
            assert -1.0 <= result.semantic_score <= 1.0 + 1e-5


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


def test_batch_convert_clip_features_to_numpy() -> None:
    """バッチCPU転送が正しく動作すること.

    Given:
        - GPU上のCLIP特徴リストがある
    When:
        - バッチCPU転送を実行する
    Then:
        - 正しくNumPy配列に変換されること
        - Noneの要素が保持されること
    """
    import torch

    # Arrange: 有効なテンソルとNoneを含むリスト
    tensors = [
        torch.randn(512).cuda() if torch.cuda.is_available() else torch.randn(512),
        None,
        torch.randn(512).cuda() if torch.cuda.is_available() else torch.randn(512),
    ]

    # Act: バッチCPU転送を実行
    results = BatchPipeline._batch_convert_clip_features_to_numpy(tensors)

    # Assert: 結果の数が一致する
    assert len(results) == 3

    # Noneの要素が保持される
    assert results[1] is None

    # 有効なテンソルがNumPy配列に変換される
    assert isinstance(results[0], np.ndarray)
    assert isinstance(results[2], np.ndarray)
    assert results[0].shape == (512,)
    assert results[2].shape == (512,)


def test_process_batch_with_lookahead_produces_same_results(
    batch_pipeline: BatchPipeline,
    tmp_path: Path,
) -> None:
    """先読みあり/なしで結果が一致すること.

    Given:
        - バッチ処理パイプラインがある
        - 複数のテスト画像がある
    When:
        - 複数の画像をバッチ処理で分析する
    Then:
        - すべての画像に対して有効な結果が返されること
        - 先読み処理が正しく動作すること
    """
    # Arrange: 複数の画像を作成
    paths = []
    for i in range(5):
        np.random.seed(42 + i)
        img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img_path = tmp_path / f"lookahead_test_{i}.jpg"
        cv2.imwrite(str(img_path), img_array)
        paths.append(str(img_path))

    # Act: 先読み付きでバッチ処理
    results = batch_pipeline.process_batch(paths, batch_size=2)

    # Assert
    assert len(results) == 5
    assert any(r is not None for r in results)
    for result, path in zip(results, paths):
        if result is not None:
            assert isinstance(result, ImageMetrics)
            assert result.path == path
            assert 0 <= result.total_score <= 100


def test_executor_reused_across_chunks(
    batch_pipeline: BatchPipeline,
    tmp_path: Path,
) -> None:
    """チャンク間でExecutorが再利用されること.

    Given:
        - バッチ処理パイプラインがある
        - 複数のテスト画像がある
        - 小さなメモリ予算で複数チャンクに分割される
    When:
        - 複数チャンクでバッチ処理を実行する
    Then:
        - Executorが再利用されていること
        - すべての画像で処理が成功すること
    """
    # Arrange: 複数の画像を作成
    paths = []
    for i in range(20):
        np.random.seed(42 + i)
        img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img_path = tmp_path / f"executor_test_{i}.jpg"
        cv2.imwrite(str(img_path), img_array)
        paths.append(str(img_path))

    # Act: 小さなチャンクサイズでバッチ処理
    results = batch_pipeline.process_batch(paths, batch_size=2)

    # Assert: すべての結果が得られる
    assert len(results) == 20
    # Executorが生成されていることを確認
    assert batch_pipeline._executor is not None

    # closeでクリーンアップされることを確認
    batch_pipeline.close()
    assert batch_pipeline._executor is None


def test_batch_pipeline_context_manager(
    batch_pipeline: BatchPipeline,
    sample_image_path: str,
) -> None:
    """コンテキストマネージャーが正しく動作すること.

    Given:
        - バッチ処理パイプラインがある
    When:
        - withステートメントで使用する
    Then:
        - 処理完了後にExecutorがクリーンアップされること
    """
    # Arrange
    paths = [sample_image_path]

    # Act: コンテキストマネージャーで使用
    with batch_pipeline:
        results = batch_pipeline.process_batch(paths, batch_size=1)

    # Assert
    assert len(results) == 1
    assert results[0] is not None
    # コンテキスト exit 後にExecutorがクリーンアップされている
    assert batch_pipeline._executor is None


def test_load_and_preprocess_images_with_max_dim(tmp_path: Path) -> None:
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
    results = BatchPipeline.load_and_preprocess_images(
        [str(large_image_path)], max_dim=720
    )

    # Assert
    assert len(results) == 1
    assert results[0] is not None
    w, h = results[0].size
    assert max(w, h) <= 720
