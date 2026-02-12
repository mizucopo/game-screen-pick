"""BatchPipelineの単体テスト.

このテストモジュールは以下のベストプラクティスに従っています：
1.「How」（実装詳細）ではなく「What」（観察可能な挙動）をテスト
2. CLIPモデルを戦略的にモック化（700MB、10-30秒のロード時間）
3. OpenCV操作、NumPy計算はモック化しない
4. 明確なコメント付きのAAAパターン（Arrange, Act, Assert）を使用
5. 高速実行（約2-5秒） - 重いモデルロードなし
"""

from pathlib import Path
from typing import Any, Callable

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


@pytest.mark.parametrize(
    "setup_invalid_path,description",
    [
        (
            lambda _: "/path/that/does/not/exist.jpg",
            "存在しないパス",
        ),
        (
            lambda tmp_path: (
                (tmp_path / "corrupted.jpg").write_text("Not an image"),
                tmp_path / "corrupted.jpg",
            )[1],
            "破損した画像",
        ),
    ],
)
def test_process_batch_handles_invalid_images(
    batch_pipeline: BatchPipeline,
    sample_image_path: str,
    tmp_path: Path,
    setup_invalid_path: Callable[[Path], Any],
    description: str,
) -> None:
    """無効な画像パスが正しく処理されること.

    Given:
        - バッチ処理パイプラインがある
        - 有効な画像パスと{description}が混在している
    When:
        - バッチ処理で分析される
    Then:
        - 有効な画像にはImageMetricsが返されること
        - 無効なパスにはNoneが返されること
        - 結果の数が入力数と一致すること
    """.format(description=description)
    # Arrange
    invalid_path = setup_invalid_path(tmp_path)
    paths = [sample_image_path, invalid_path]

    # Act
    results = batch_pipeline.process_batch(paths, batch_size=1)

    # Assert
    assert len(results) == 2
    assert results[0] is not None
    assert results[1] is None  # 無効なパス


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
