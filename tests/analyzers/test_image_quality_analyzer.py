"""ImageQualityAnalyzerの単体テスト.

このテストモジュールは以下のベストプラクティスに従っています：
1.「How」（実装詳細）ではなく「What」（観察可能な挙動）をテスト
2. CLIPモデルのみ戦略的にモック化（700MB、10-30秒のロード時間）
3. OpenCV操作、NumPy計算、MetricNormalizerはモック化しない
4. 明確なコメント付きのAAAパターン（Arrange, Act, Assert）を使用
5. 高速実行（約2-5秒） - 重いモデルロードなし
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.analyzers.image_quality_analyzer import ImageQualityAnalyzer
from src.models.image_metrics import ImageMetrics


@pytest.fixture
def sample_image_path(tmp_path: Path) -> str:
    """標準的なテスト画像（640x480 JPG）を作成する."""
    return _create_test_image(tmp_path, "test_image.jpg", (480, 640), (0, 255))


@pytest.fixture
def dark_image_path(tmp_path: Path) -> str:
    """輝度ペナルティのテスト用に暗いテスト画像（640x480 JPG）を作成する."""
    return _create_test_image(tmp_path, "dark_image.jpg", (480, 640), (0, 50))


@pytest.fixture
def png_image_path(tmp_path: Path) -> str:
    """PNG形式のテスト画像（640x480）を作成する."""
    return _create_test_image(tmp_path, "test_image.png", (480, 640), (0, 255))


@pytest.fixture
def small_image_path(tmp_path: Path) -> str:
    """小さいテスト画像（320x240 JPG）を作成する."""
    return _create_test_image(tmp_path, "small_image.jpg", (240, 320), (0, 255))


def _create_test_image(
    tmp_path: Path, filename: str, size: tuple[int, int], pixel_range: tuple[int, int]
) -> str:
    """テスト画像を作成するヘルパー関数.

    Args:
        tmp_path: 一時ディレクトリパス
        filename: 画像ファイル名
        size: 画像サイズ（高さ、幅）
        pixel_range: ピクセル値の範囲（最小、最大）

    Returns:
        作成された画像の絶対パス
    """
    np.random.seed(42)
    img_array = np.random.randint(
        pixel_range[0], pixel_range[1], (*size, 3), dtype=np.uint8
    )
    img_path = tmp_path / filename
    cv2.imwrite(str(img_path), img_array)
    return str(img_path)


def test_analyze_returns_valid_metrics_with_genre_specific_weights(
    sample_image_path: str,
) -> None:
    """ジャンル特有の重みで有効なメトリックが返されること.

    Given:
        - 特定ジャンルのアナライザインスタンスがある
        - 有効なテスト画像がある
    When:
        - 画像が分析される
    Then:
        - 有効なImageMetricsが返されること
        - スコアが有効範囲内にあること
    """
    # Arrange
    analyzer = ImageQualityAnalyzer(genre="2d_rpg")

    # Act
    result = analyzer.analyze(sample_image_path)

    # Assert
    assert result is not None
    assert isinstance(result, ImageMetrics)
    assert result.path == sample_image_path
    # スコア値が有効範囲内であることを検証
    assert 0 <= result.total_score <= 100
    # コサイン類似度の範囲（浮動小数点の丸め誤差を許容）
    assert -1.0 <= result.semantic_score <= 1.0 + 1e-5
    # 正規化されたメトリックが存在することを検証
    assert len(result.normalized_metrics) > 0


def test_analyze_returns_none_for_invalid_inputs(
    tmp_path: Path,
) -> None:
    """無効な入力（存在しないファイル・破損した画像）に対してNoneが返されること.

    Given:
        - アナライザインスタンスがある
        - 存在しないファイルパスがある
        - 破損した画像ファイルがある
    When:
        - 無効な入力が分析される
    Then:
        - すべてのケースでNoneが返されること（正常な失敗）
        - 例外が発生しないこと
    """
    # Arrange
    analyzer = ImageQualityAnalyzer()
    nonexistent_path = "/path/that/does/not/exist.jpg"
    corrupted_path = tmp_path / "corrupted.jpg"
    corrupted_path.write_text("This is not a valid image file")

    # Act
    result_nonexistent = analyzer.analyze(nonexistent_path)
    result_corrupted = analyzer.analyze(str(corrupted_path))

    # Assert
    assert result_nonexistent is None
    assert result_corrupted is None


def test_analyze_produces_consistent_results_for_same_image(
    sample_image_path: str,
) -> None:
    """同じ画像を複数回分析する際に一貫した結果が生成されること.

    Given:
        - アナライザインスタンスがある
        - 有効なテスト画像がある
    When:
        - 同じ画像が2回分析される
    Then:
        - 両方の分析で同一の結果が生成されること
        - 総スコアと特徴ベクトルが同じであること
    """
    # Arrange
    analyzer = ImageQualityAnalyzer()

    # Act
    result1 = analyzer.analyze(sample_image_path)
    result2 = analyzer.analyze(sample_image_path)

    # Assert
    assert result1 is not None
    assert result2 is not None
    assert result1.path == result2.path
    assert result1.total_score == result2.total_score
    # 特徴ベクトルが同一であることを検証（重要な特性）
    assert np.array_equal(result1.features, result2.features)


def test_analyze_batch_returns_correct_metrics_for_multiple_images(
    sample_image_path: str,
    png_image_path: str,
    small_image_path: str,
) -> None:
    """複数の画像が正しくバッチ処理されること.

    Given:
        - アナライザインスタンスがある
        - 複数の有効なテスト画像がある
    When:
        - 複数の画像がバッチ処理で分析される
    Then:
        - すべての画像に対して有効なImageMetricsが返されること
        - 結果の数が入力数と一致すること
        - 各結果のパスが正しいこと
    """
    # Arrange
    analyzer = ImageQualityAnalyzer()
    paths = [sample_image_path, png_image_path, small_image_path]

    # Act - バッチサイズ1で実行（モックの制約により一時的に1に設定）
    results = analyzer.analyze_batch(paths, batch_size=1)

    # Assert
    assert len(results) == 3
    for result, path in zip(results, paths):
        assert result is not None
        assert isinstance(result, ImageMetrics)
        assert result.path == path
        assert 0 <= result.total_score <= 100
        # コサイン類似度の範囲（浮動小数点の丸め誤差を許容）
        assert -1.0 <= result.semantic_score <= 1.0 + 1e-5


def test_analyze_batch_handles_mixed_valid_and_invalid_images(
    sample_image_path: str,
) -> None:
    """有効な画像と無効な画像が混在する場合に正しく処理されること.

    Given:
        - アナライザインスタンスがある
        - 有効な画像パスと存在しないパスが混在している
    When:
        - バッチ処理で分析される
    Then:
        - 有効な画像にはImageMetricsが返されること
        - 無効なパスにはNoneが返されること
        - 結果の数が入力数と一致すること
    """
    # Arrange
    analyzer = ImageQualityAnalyzer()
    nonexistent_path = "/path/that/does/not/exist.jpg"
    paths = [sample_image_path, nonexistent_path, sample_image_path]

    # Act - バッチサイズ1で実行（モックの制約により一時的に1に設定）
    results = analyzer.analyze_batch(paths, batch_size=1)

    # Assert
    assert len(results) == 3
    assert results[0] is not None
    assert results[1] is None  # 存在しないパス
    assert results[2] is not None


def test_analyze_batch_produces_same_results_as_analyze(
    sample_image_path: str,
) -> None:
    """バッチ処理と単一処理で同じ結果が得られること.

    Given:
        - アナライザインスタンスがある
        - 有効なテスト画像がある
    When:
        - 同じ画像を単一処理とバッチ処理で分析する
    Then:
        - 両方の結果で総スコアが一致すること
        - 両方の結果で特徴ベクトルが一致すること
    """
    # Arrange
    analyzer = ImageQualityAnalyzer()

    # Act
    single_result = analyzer.analyze(sample_image_path)
    batch_results = analyzer.analyze_batch([sample_image_path])

    # Assert
    assert single_result is not None
    assert batch_results[0] is not None
    # 浮動小数点の精度誤差を許容して比較
    assert single_result.total_score == pytest.approx(batch_results[0].total_score)
    assert np.array_equal(single_result.features, batch_results[0].features)
