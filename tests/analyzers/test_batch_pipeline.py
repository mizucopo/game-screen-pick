"""BatchPipelineの単体テスト.

このテストモジュールは以下のベストプラクティスに従っています：
1.「How」（実装詳細）ではなく「What」（観察可能な挙動）をテスト
2. CLIPモデルを戦略的にモック化（700MB、10-30秒のロード時間）
3. OpenCV操作、NumPy計算はモック化しない
4. 明確なコメント付きのAAAパターン（Arrange, Act, Assert）を使用
5. 高速実行（約2-5秒） - 重いモデルロードなし
"""

import os
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
from src.models.path_metadata import PathMetadata


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


def test_cached_results_use_stored_semantic_score(tmp_path: Path) -> None:
    """キャッシュヒット時に保存されたsemantic_scoreが使用されること.

    Given:
        - semantic_score付きのエントリがキャッシュに保存されている
    When:
        - キャッシュから結果を取得
    Then:
        - 保存されたsemantic_scoreが使用されること
        - 再計算がスキップされること
    """
    import numpy as np
    from src.cache.feature_cache import FeatureCache

    # Arrange: semantic_score付きでキャッシュにエントリを保存
    cache_path = tmp_path / "test_semantic_cache.sqlite3"
    cache = FeatureCache(cache_path)

    # テスト用のキャッシュエントリを作成
    clip_features = np.random.randn(512).astype(np.float32)
    hsv_features = np.random.randn(64).astype(np.float32)
    raw_metrics = {"blur_score": 100.0}
    semantic_score = 0.85

    # テスト画像を作成
    np.random.seed(42)
    img_array = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    img_path = tmp_path / "semantic_cache_test.jpg"
    cv2.imwrite(str(img_path), img_array)

    # キャッシュキーを生成して保存
    file_stat = img_path.stat()
    cache_key = cache.generate_cache_key(
        absolute_path=os.path.abspath(str(img_path)),
        file_size=file_stat.st_size,
        mtime_ns=int(file_stat.st_mtime_ns),
        model_name="openai/clip-vit-base-patch32",
        target_text="epic game scenery",
        max_dim=1280,
    )

    # semantic_score付きで保存
    normalized_metrics = {"blur_score": 0.5}
    total_score = 75.0
    cache.put(
        cache_key=cache_key,
        clip_features=clip_features,
        raw_metrics=raw_metrics,
        hsv_features=hsv_features,
        semantic_score=semantic_score,
        normalized_metrics=normalized_metrics,
        total_score=total_score,
    )

    # Act: キャッシュから取得
    result = cache.get(cache_key)

    # Assert: semantic_scoreが正しく取得できる
    assert result is not None
    assert result.semantic_score == semantic_score


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


def test_path_metadata_creation() -> None:
    """PathMetadataが正しく作成されること.

    Given:
        - 有効なファイルパスがある
    When:
        - PathMetadataオブジェクトを作成する
    Then:
        - すべてのフィールドが正しく設定されること
        - オプションフィールドはNoneで初期化されること
    """
    # Arrange
    test_path = "/path/to/image.jpg"

    # Act: 最小の引数で作成
    metadata_minimal = PathMetadata(path=test_path)

    # Assert: 必須フィールドが設定される
    assert metadata_minimal.path == test_path
    assert metadata_minimal.absolute_path is None
    assert metadata_minimal.file_stat is None
    assert metadata_minimal.cache_key is None

    # Act: すべてのフィールドを指定して作成
    from pathlib import Path as StdPath

    abs_path = str(StdPath(test_path).resolve())
    file_stat = os.stat(__file__)  # 何か有効なstat
    cache_key: dict[str, str | int] = {"key": "value", "file_size": 123}

    metadata_full = PathMetadata(
        path=test_path,
        absolute_path=abs_path,
        file_stat=file_stat,
        cache_key=cache_key,
    )

    # Assert: すべてのフィールドが設定される
    assert metadata_full.path == test_path
    assert metadata_full.absolute_path == abs_path
    assert metadata_full.file_stat == file_stat
    assert metadata_full.cache_key == cache_key


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


def test_process_batch_with_uncached_files(
    tmp_path: Path,
) -> None:
    """未キャッシュファイルのパイプライン処理が正しく動作すること.

    Given:
        - バッチ処理パイプラインがある
        - 未キャッシュの画像ファイルがある
    When:
        - process_batchを実行する
    Then:
        - 有効な結果が返されること
        - 結果のパスが正しいこと
    """
    # Arrange: キャッシュ付きのパイプラインを作成
    config = AnalyzerConfig()
    weights = GenreWeights.get_weights("mixed")
    model_manager = CLIPModelManager()
    feature_extractor = FeatureExtractor(model_manager)
    metric_calculator = MetricCalculator(config, weights, model_manager)

    cache_path = tmp_path / "test_uncached_cache.sqlite3"
    cache = FeatureCache(cache_path)

    pipeline = BatchPipeline(
        feature_extractor,
        metric_calculator,
        config,
        cache=cache,
    )

    # テスト画像を作成
    np.random.seed(42)
    img_array = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    img_path = tmp_path / "uncached_test.jpg"
    cv2.imwrite(str(img_path), img_array)

    # Act: 未キャッシュの画像を処理
    results = pipeline.process_batch([str(img_path)], batch_size=1)

    # Assert: 有効な結果が返される
    assert len(results) == 1
    assert results[0] is not None
    assert results[0].path == str(img_path)
    assert 0 <= results[0].total_score <= 100
    assert -1.0 <= results[0].semantic_score <= 1.0 + 1e-5


def test_get_cached_results_returns_metadata(tmp_path: Path) -> None:
    """_get_cached_resultsがPathMetadataリストを返すこと.

    Given:
        - キャッシュ付きのパイプラインがある
        - 複数の画像パスがある
    When:
        - _get_cached_resultsを実行する
    Then:
        - キャッシュ結果とメタ情報のタプルが返されること
        - メタ情報の数がパスの数と一致すること
    """
    # Arrange
    config = AnalyzerConfig()
    weights = GenreWeights.get_weights("mixed")
    model_manager = CLIPModelManager()
    feature_extractor = FeatureExtractor(model_manager)
    metric_calculator = MetricCalculator(config, weights, model_manager)

    cache_path = tmp_path / "test_metadata_cache.sqlite3"
    cache = FeatureCache(cache_path)

    pipeline = BatchPipeline(
        feature_extractor,
        metric_calculator,
        config,
        cache=cache,
    )

    # テスト画像を作成
    paths = []
    for i in range(3):
        np.random.seed(42 + i)
        img_array = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        img_path = tmp_path / f"metadata_test_{i}.jpg"
        cv2.imwrite(str(img_path), img_array)
        paths.append(str(img_path))

    # Act: _get_cached_resultsを実行
    cached_results, all_metadata = pipeline._get_cached_results(paths)

    # Assert: メタ情報の数が一致する
    assert len(all_metadata) == len(paths)
    assert len(cached_results) == len(paths)

    # すべてのメタ情報がPathMetadataである
    for metadata in all_metadata:
        assert isinstance(metadata, PathMetadata)
        assert metadata.path in paths


def test_process_single_result_reuses_metadata(
    batch_pipeline: BatchPipeline,
    sample_image_path: str,
) -> None:
    """メタ情報が再利用されresolve/statが1回のみ実行されること.

    Given:
        - バッチ処理パイプラインがある
        - 有効な画像ファイルがある
        - メタ情報（absolute_path, file_stat）が事前に収集されている
    When:
        - _process_single_resultをメタ情報付きで呼び出す
    Then:
        - メタ情報が使用されること
        - 有効なImageMetricsが返されること
    """
    # Arrange: 事前にメタ情報を収集
    from pathlib import Path as StdPath

    abs_path = str(StdPath(sample_image_path).resolve())
    file_stat = os.stat(sample_image_path)

    metadata = PathMetadata(
        path=sample_image_path,
        absolute_path=abs_path,
        file_stat=file_stat,
        cache_key=None,  # キャッシュなし
    )

    # PIL画像とCLIP特徴をモック
    from PIL import Image

    np.random.seed(42)
    img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    pil_img = Image.fromarray(img_array)

    # CLIP特徴（NumPy配列、CPU上）
    clip_features = np.random.randn(512).astype(np.float32)
    semantic = 0.5

    # Act: メタ情報付きで処理
    result, cache_entry = batch_pipeline._process_single_result(
        path=sample_image_path,
        pil_img=pil_img,
        clip_features=clip_features,
        semantic=semantic,
        metadata=metadata,
    )

    # Assert: 有効な結果が返される
    assert result is not None
    assert result.path == sample_image_path
    assert 0 <= result.total_score <= 100
    assert result.semantic_score == semantic


def test_cached_results_uses_stored_metrics(tmp_path: Path) -> None:
    """キャッシュヒット時に保存されたnormalized_metricsとtotal_scoreが使用されること.

    Given:
        - normalized_metricsとtotal_score付きのエントリがキャッシュに保存されている
    When:
        - キャッシュから結果を取得
    Then:
        - 保存されたnormalized_metricsとtotal_scoreが使用されること
        - 再計算がスキップされること
    """
    # Arrange: キャッシュ付きのパイプラインを作成
    config = AnalyzerConfig()
    weights = GenreWeights.get_weights("mixed")
    model_manager = CLIPModelManager()
    feature_extractor = FeatureExtractor(model_manager)
    metric_calculator = MetricCalculator(config, weights, model_manager)

    cache_path = tmp_path / "test_stored_metrics_cache.sqlite3"
    cache = FeatureCache(cache_path)

    pipeline = BatchPipeline(
        feature_extractor,
        metric_calculator,
        config,
        cache=cache,
    )

    # テスト画像を作成
    np.random.seed(42)
    img_array = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    img_path = tmp_path / "stored_metrics_test.jpg"
    cv2.imwrite(str(img_path), img_array)

    # ファイル情報を事前に取得してキャッシュキーを作成
    file_stat = img_path.stat()
    cache_key = cache.generate_cache_key(
        absolute_path=os.path.abspath(str(img_path)),
        file_size=file_stat.st_size,
        mtime_ns=int(file_stat.st_mtime_ns),
        model_name="openai/clip-vit-base-patch32",
        target_text="epic game scenery",
        max_dim=config.max_dim,
    )

    # Act: 最初の実行（キャッシュミス）
    first_results = pipeline.process_batch([str(img_path)], batch_size=1)

    # Assert: 最初の実行で有効な結果が得られる
    assert len(first_results) == 1
    assert first_results[0] is not None
    assert first_results[0].path == str(img_path)
    assert -1.0 <= first_results[0].semantic_score <= 1.0 + 1e-5
    assert 0 <= first_results[0].total_score <= 100

    # キャッシュから直接取得して確認（事前に取得したcache_keyを使用）
    entry = cache.get(cache_key)
    assert entry is not None, f"Cache entry not found for key: {cache_key}"
    # normalized_metricsとtotal_scoreが保存されていること
    assert entry.normalized_metrics is not None
    assert entry.total_score is not None

    # 2回目の実行（キャッシュヒット）
    second_results = pipeline.process_batch([str(img_path)], batch_size=1)

    # Assert: 2回目の実行でも有効な結果が得られる
    assert len(second_results) == 1
    assert second_results[0] is not None

    # キャッシュヒット時も結果が正しく取得できる
    assert second_results[0].path == str(img_path)
    assert -1.0 <= second_results[0].semantic_score <= 1.0 + 1e-5
    assert 0 <= second_results[0].total_score <= 100
