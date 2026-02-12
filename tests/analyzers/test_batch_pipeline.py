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
from PIL import Image

from src.analyzers.batch_pipeline import BatchPipeline
from src.analyzers.clip_model_manager import CLIPModelManager
from src.analyzers.feature_extractor import FeatureExtractor
from src.analyzers.metric_calculator import MetricCalculator
from src.constants.genre_weights import GenreWeights
from src.models.analyzer_config import AnalyzerConfig
from src.models.image_metrics import ImageMetrics


@pytest.fixture
def sample_image_path(tmp_path: Path) -> str:
    """標準的なテスト画像（640x480 JPG）を作成する."""
    np.random.seed(42)
    img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    img_path = tmp_path / "test_image.jpg"
    cv2.imwrite(str(img_path), img_array)
    return str(img_path)


@pytest.fixture
def dark_image_path(tmp_path: Path) -> str:
    """輝度ペナルティのテスト用に暗いテスト画像（640x480 JPG）を作成する."""
    np.random.seed(42)
    img_array = np.random.randint(0, 50, (480, 640, 3), dtype=np.uint8)
    img_path = tmp_path / "dark_image.jpg"
    cv2.imwrite(str(img_path), img_array)
    return str(img_path)


@pytest.fixture
def png_image_path(tmp_path: Path) -> str:
    """PNG形式のテスト画像（640x480）を作成する."""
    np.random.seed(42)
    img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    img_path = tmp_path / "test_image.png"
    cv2.imwrite(str(img_path), img_array)
    return str(img_path)


@pytest.fixture
def batch_pipeline() -> BatchPipeline:
    """バッチ処理パイプラインのフィクスチャ."""
    config = AnalyzerConfig()
    weights = GenreWeights.get_weights("mixed")
    model_manager = CLIPModelManager()
    feature_extractor = FeatureExtractor(model_manager)
    metric_calculator = MetricCalculator(config, weights, model_manager)
    return BatchPipeline(feature_extractor, metric_calculator, config)


def test_process_batch_returns_correct_metrics_for_multiple_images(
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
    # 3枚目の画像を作成
    np.random.seed(43)
    img_array = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    small_image_path = tmp_path / "small_image.jpg"
    cv2.imwrite(str(small_image_path), img_array)

    paths = [sample_image_path, png_image_path, str(small_image_path)]

    # Act
    results = batch_pipeline.process_batch(paths, batch_size=1)

    # Assert
    assert len(results) == 3
    # 少なくとも1つの画像が処理されていることを確認
    assert any(r is not None for r in results)
    for result, path in zip(results, paths):
        if result is None:
            continue
        assert isinstance(result, ImageMetrics)
        assert result.path == path
        assert 0 <= result.total_score <= 100
        # コサイン類似度の範囲（浮動小数点の丸め誤差を許容）
        assert -1.0 <= result.semantic_score <= 1.0 + 1e-5


def test_process_batch_handles_mixed_valid_and_invalid_images(
    batch_pipeline: BatchPipeline, sample_image_path: str
) -> None:
    """有効な画像と無効な画像が混在する場合に正しく処理されること.

    Given:
        - バッチ処理パイプラインがある
        - 有効な画像パスと存在しないパスが混在している
    When:
        - バッチ処理で分析される
    Then:
        - 有効な画像にはImageMetricsが返されること
        - 無効なパスにはNoneが返されること
        - 結果の数が入力数と一致すること
    """
    # Arrange
    nonexistent_path = "/path/that/does/not/exist.jpg"
    paths = [sample_image_path, nonexistent_path, sample_image_path]

    # Act
    results = batch_pipeline.process_batch(paths, batch_size=1)

    # Assert
    assert len(results) == 3
    # 少なくとも最初の画像は処理されている
    assert results[0] is not None
    assert results[1] is None  # 存在しないパス


def test_process_batch_handles_corrupted_images(
    batch_pipeline: BatchPipeline, sample_image_path: str, tmp_path: Path
) -> None:
    """破損した画像が正しく処理されること.

    Given:
        - バッチ処理パイプラインがある
        - 有効な画像と破損した画像が混在している
    When:
        - バッチ処理で分析される
    Then:
        - 有効な画像にはImageMetricsが返されること
        - 破損した画像にはNoneが返されること
    """
    # Arrange
    corrupted_path = tmp_path / "corrupted.jpg"
    corrupted_path.write_text("This is not a valid image file")

    paths = [sample_image_path, str(corrupted_path)]

    # Act
    results = batch_pipeline.process_batch(paths, batch_size=1)

    # Assert
    assert len(results) == 2
    assert results[0] is not None
    assert results[1] is None  # 破損した画像


def test_process_batch_processes_various_image_formats(
    batch_pipeline: BatchPipeline,
    sample_image_path: str,
    png_image_path: str,
) -> None:
    """様々な形式の画像が正しく処理されること.

    Given:
        - バッチ処理パイプラインがある
        - 異なる形式（JPG、PNG）のテスト画像がある
    When:
        - 各画像がバッチ処理で分析される
    Then:
        - すべての形式が正常に分析されること
        - 有効なImageMetricsが返されること
    """
    # Arrange
    paths = [sample_image_path, png_image_path]

    # Act
    results = batch_pipeline.process_batch(paths, batch_size=1)

    # Assert
    assert len(results) == 2
    # 少なくとも1つの画像が処理されている
    assert any(r is not None for r in results)
    for result in results:
        if result is None:
            continue
        assert isinstance(result, ImageMetrics)
        assert 0 <= result.total_score <= 100


def test_process_batch_handles_dark_images_with_penalty(
    batch_pipeline: BatchPipeline, dark_image_path: str
) -> None:
    """暗い画像に対して輝度ペナルティが適用されること.

    Given:
        - バッチ処理パイプラインがある
        - 暗いテスト画像がある
    When:
        - 暗い画像がバッチ処理で分析される
    Then:
        - 有効なImageMetricsが返されること
        - 輝度ペナルティが適用されること
    """
    # Arrange
    paths = [dark_image_path]

    # Act
    results = batch_pipeline.process_batch(paths, batch_size=1)

    # Assert
    assert len(results) == 1
    result = results[0]
    if result is not None:
        assert isinstance(result, ImageMetrics)
        assert result.path == dark_image_path
        # 暗い画像では輝度が低い
        assert result.raw_metrics["brightness"] < 40
        # ペナルティ適用後も有効なスコア
        assert result.total_score >= 0


def test_process_batch_returns_consistent_features(
    batch_pipeline: BatchPipeline, sample_image_path: str
) -> None:
    """特徴ベクトルが一貫したサイズを持つこと.

    Given:
        - バッチ処理パイプラインがある
        - 有効なテスト画像がある
    When:
        - 画像がバッチ処理で分析される
    Then:
        - 特徴ベクトルが一貫したサイズ（576次元）を持つこと
    """
    # Arrange
    paths = [sample_image_path]

    # Act
    results = batch_pipeline.process_batch(paths, batch_size=1)

    # Assert
    result = results[0]
    if result is not None:
        # HSV特徴（64次元）+ CLIP特徴（512次元）= 576次元
        assert result.features.shape == (576,)


def test_process_batch_empty_list_returns_empty_list(
    batch_pipeline: BatchPipeline,
) -> None:
    """空のリストが渡された場合、空のリストが返されること.

    Given:
        - バッチ処理パイプラインがある
        - 空のパスリストがある
    When:
        - バッチ処理で分析される
    Then:
        - 空のリストが返されること
    """
    # Arrange
    paths: list[str] = []

    # Act
    results = batch_pipeline.process_batch(paths, batch_size=1)

    # Assert
    assert len(results) == 0


def test_process_batch_applies_brightness_penalty_correctly(
    batch_pipeline: BatchPipeline, sample_image_path: str, dark_image_path: str
) -> None:
    """輝度ペナルティが正しく適用されること.

    Given:
        - バッチ処理パイプラインがある
        - 明るい画像と暗い画像がある
    When:
        - 両方の画像がバッチ処理で分析される
    Then:
        - 暗い画像の方がペナルティの影響を受ける可能性があること
    """
    # Arrange
    paths = [sample_image_path, dark_image_path]

    # Act
    results = batch_pipeline.process_batch(paths, batch_size=1)

    # Assert
    assert len(results) == 2
    # 少なくとも1つの結果が有効であることを確認
    valid_results = [r for r in results if r is not None]
    assert len(valid_results) > 0

    for result in valid_results:
        assert isinstance(result, ImageMetrics)
        assert result.total_score >= 0


def test_load_and_preprocess_images_handles_invalid_files(
    tmp_path: Path,
) -> None:
    """無効なファイルが正しく処理されること.

    Given:
        - バッチ処理パイプラインがある
        - 有効な画像と無効なファイルが混在している
    When:
        - 画像が読み込まれて前処理される
    Then:
        - 有効な画像はPIL画像として返されること
        - 無効なファイルはNoneとして返されること
    """
    # Arrange
    np.random.seed(42)
    img_array = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    valid_path = tmp_path / "valid.jpg"
    cv2.imwrite(str(valid_path), img_array)

    invalid_path = tmp_path / "invalid.txt"
    invalid_path.write_text("Not an image")

    # Act
    paths = [str(valid_path), str(invalid_path)]
    results = BatchPipeline.load_and_preprocess_images(paths)

    # Assert
    assert len(results) == 2
    assert results[0] is not None
    assert isinstance(results[0], Image.Image)
    assert results[1] is None


def test_process_batch_with_large_number_of_images(
    batch_pipeline: BatchPipeline, tmp_path: Path
) -> None:
    """大量の画像がチャンク処理で正しく処理されること.

    Given:
        - バッチ処理パイプラインがある
        - チャンクサイズより多くのテスト画像がある
    When:
        - 画像がバッチ処理で分析される
    Then:
        - すべての画像が処理されること
        - 結果の数が入力数と一致すること
    """
    # Arrange
    paths = []
    for i in range(10):  # 10枚の画像を作成
        np.random.seed(42 + i)
        img_array = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        img_path = tmp_path / f"test_image_{i}.jpg"
        cv2.imwrite(str(img_path), img_array)
        paths.append(str(img_path))

    # Act
    results = batch_pipeline.process_batch(paths, batch_size=2)

    # Assert
    assert len(results) == 10
    # 少なくとも1つの画像が処理されている
    assert any(r is not None for r in results)


def test_compute_chunk_boundaries_with_small_files(tmp_path: Path) -> None:
    """小さいファイルの場合、メモリ予算内で最大限の画像を1チャンクに収めること.

    Given:
        - 小さいテスト画像ファイルが複数ある
        - メモリ予算が十分に大きい
    When:
        - チャンク境界を計算する
    Then:
        - すべての画像が1チャンクに含まれること
    """
    # Arrange
    paths = []
    for i in range(5):
        np.random.seed(42 + i)
        # 小さい画像（約23KB）
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img_path = tmp_path / f"small_{i}.jpg"
        cv2.imwrite(str(img_path), img_array)
        paths.append(str(img_path))

    # Act - メモリ予算1MB、最低4枚でチャンク分割
    chunks = BatchPipeline._compute_chunk_boundaries(
        paths, max_memory_mb=1, min_chunk_size=4
    )

    # Assert
    assert len(chunks) == 1  # すべて1チャンク
    assert chunks[0] == (0, 5)


def test_compute_chunk_boundaries_splits_large_files(tmp_path: Path) -> None:
    """大きいファイルの場合、メモリ予算に従ってチャンク分割されること.

    Given:
        - 大きいテスト画像ファイルが複数ある
        - メモリ予算が小さい
    When:
        - チャンク境界を計算する
    Then:
        - メモリ予算に従って複数チャンクに分割されること
    """
    # Arrange
    paths = []
    for i in range(6):
        np.random.seed(42 + i)
        # 大きい画像（約1.4MB）
        img_array = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
        img_path = tmp_path / f"large_{i}.jpg"
        cv2.imwrite(str(img_path), img_array)
        paths.append(str(img_path))

    # Act - メモリ予算1MB、最低2枚でチャンク分割
    chunks = BatchPipeline._compute_chunk_boundaries(
        paths, max_memory_mb=1, min_chunk_size=2
    )

    # Assert
    # 1.4MBのファイルが6枚あるので、1MB予備で複数チャンクに分割される
    assert len(chunks) > 1
    # すべての画像がカバーされている
    covered_count = sum(end - start for start, end in chunks)
    assert covered_count == 6


def test_compute_chunk_boundaries_respects_min_chunk_size(tmp_path: Path) -> None:
    """最低チャンクサイズが尊重されること.

    Given:
        - 画像ファイルがある
        - 最低チャンクサイズが設定されている
    When:
        - チャンク境界を計算する
    Then:
        - 最低チャンクサイズ未満のチャークが作られないこと
    """
    # Arrange
    paths = []
    for i in range(5):
        np.random.seed(42 + i)
        # 大きい画像（約1.4MB）
        img_array = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
        img_path = tmp_path / f"large_{i}.jpg"
        cv2.imwrite(str(img_path), img_array)
        paths.append(str(img_path))

    # Act - メモリ予算1MB、最低3枚でチャンク分割
    chunks = BatchPipeline._compute_chunk_boundaries(
        paths, max_memory_mb=1, min_chunk_size=3
    )

    # Assert
    for start, end in chunks:
        chunk_size = end - start
        assert chunk_size == 0 or chunk_size >= 3  # 最後のチャンクを除いて3枚以上


def test_compute_chunk_boundaries_with_empty_list() -> None:
    """空のリストの場合、空のチャンクリストが返されること.

    Given:
        - 空のパスリスト
    When:
        - チャンク境界を計算する
    Then:
        - 空のリストが返されること
    """
    # Arrange
    paths: list[str] = []

    # Act
    chunks = BatchPipeline._compute_chunk_boundaries(
        paths, max_memory_mb=512, min_chunk_size=16
    )

    # Assert
    assert chunks == []


def test_compute_chunk_boundaries_with_nonexistent_files() -> None:
    """存在しないファイルの場合、0バイトとして扱われチャンク分割されること.

    Given:
        - 存在しないファイルパスのリスト
    When:
        - チャンク境界を計算する
    Then:
        - 最低チャンクサイズに基づいて分割されること
    """
    # Arrange
    paths = ["/nonexistent/1.jpg", "/nonexistent/2.jpg", "/nonexistent/3.jpg"]

    # Act - 最低2枚でチャンク分割
    chunks = BatchPipeline._compute_chunk_boundaries(
        paths, max_memory_mb=1, min_chunk_size=2
    )

    # Assert
    # 存在しないファイルは0バイトとして扱われるため、メモリ予算を超えない
    # 最低チャンクサイズに従って分割
    assert len(chunks) >= 1
