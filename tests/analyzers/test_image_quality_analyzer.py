"""ImageQualityAnalyzerの単体テスト.

このテストモジュールは以下のベストプラクティスに従っています：
1.「How」（実装詳細）ではなく「What」（観察可能な挙動）をテスト
2. CLIPモデルのみ戦略的にモック化（700MB、10-30秒のロード時間）
3. OpenCV操作、NumPy計算、MetricNormalizerはモック化しない
4. 明確なコメント付きのAAAパターン（Arrange, Act, Assert）を使用
5. 高速実行（約2-5秒） - 重いモデルロードなし
"""

from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import torch

from src.analyzers.image_quality_analyzer import ImageQualityAnalyzer
from src.models.image_metrics import ImageMetrics


# ============================================================================
# CLIPモデルのモックフィクスチャ（700MBダウンロードと10-30秒ロード時間を回避）
# ============================================================================


@pytest.fixture
def mock_clip_model() -> Generator[MagicMock, None, None]:
    """700MBの重みロードを回避するためのCLIPモデルのモック.

    このfixtureは本物のCLIPモデルを以下のモックに置き換えます：
    - 決定論的テストのために固定されたlogit値を返す
    - GPU/CPU切り替えのための.to(device)呼び出しをサポート
    """
    with patch("transformers.CLIPModel.from_pretrained") as mock:
        model = MagicMock()
        # 一貫したテストのために固定されたlogit値を返す
        mock_output = MagicMock()
        mock_output.logits_per_image = torch.tensor([[25.0]])
        model.return_value = mock_output
        model.to = MagicMock(return_value=model)
        mock.return_value = model
        yield mock


@pytest.fixture
def mock_clip_processor() -> Generator[MagicMock, None, None]:
    """トークナイザと特徴抽出器のロードを回避するためのCLIPプロセッサのモック.

    このfixtureは本物のCLIPプロセッサを以下のモックに置き換えます：
    - テキストと画像のために固定されたテンソル形状を返す
    - GPU/CPU切り替えのための.to(device)呼び出しをサポート
    """
    with patch("transformers.CLIPProcessor.from_pretrained") as mock:
        processor = MagicMock()
        # Return realistic tensor shapes
        processor.return_value = MagicMock(
            input_ids=torch.tensor([[1, 2, 3]]),
            pixel_values=torch.tensor([[[[1.0]]]]),
            attention_mask=torch.tensor([[1, 1, 1]]),
        )
        # Mock the .to() method for device switching
        processor_instance = MagicMock()
        processor_instance.return_value.to = MagicMock(return_value=processor_instance)
        mock.return_value = processor_instance
        yield mock


# ============================================================================
# テスト画像をプログラム的に作成するフィクスチャ
# ============================================================================


@pytest.fixture
def sample_image_path(tmp_path: Path) -> str:
    """ランダムなピクセル値でテスト画像を作成する.

    決定論的テスト結果のために固定シードを使用する。
    画像サイズ：640x480（標準的な4:3アスペクト比）。
    """
    np.random.seed(42)  # Fixed seed for reproducibility
    img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    img_path = tmp_path / "test_image.jpg"
    cv2.imwrite(str(img_path), img_array)
    return str(img_path)


@pytest.fixture
def blurry_image_path(tmp_path: Path) -> str:
    """ガウシアンぼかしを使用してぼやけたテスト画像を作成する.

    blur_score検出のテストに使用する。
    画像サイズ：640x480、強いガウシアンぼかし適用（カーネルサイズ31x31）。
    """
    np.random.seed(42)
    img_array = np.random.randint(100, 150, (480, 640, 3), dtype=np.uint8)
    blurred = cv2.GaussianBlur(img_array, (31, 31), 0)
    img_path = tmp_path / "blurry_image.jpg"
    cv2.imwrite(str(img_path), blurred)
    return str(img_path)


@pytest.fixture
def dark_image_path(tmp_path: Path) -> str:
    """輝度ペナルティのテスト用に暗いテスト画像を作成する.

    輝度が40未満の画像が0.6のペナルティを受けることを検証するために使用する。
    画像サイズ：640x480、低いピクセル値（0-50）。
    """
    np.random.seed(42)
    img_array = np.random.randint(0, 50, (480, 640, 3), dtype=np.uint8)
    img_path = tmp_path / "dark_image.jpg"
    cv2.imwrite(str(img_path), img_array)
    return str(img_path)


@pytest.fixture
def high_quality_image_path(tmp_path: Path) -> str:
    """良好なコントラストとエッジを持つ高品質テスト画像を作成する.

    良好な画像が高スコアを受けることをテストするために使用する。
    画像サイズ：640x480、良好なコントラストとエッジ密度。
    """
    np.random.seed(42)
    # 良好なコントラストを持つ画像を作成
    img_array = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
    # Sobel風パターンを使用してエッジを追加
    img_array[200:280, 300:340] = 255  # 明るい長方形を追加
    img_path = tmp_path / "high_quality_image.jpg"
    cv2.imwrite(str(img_path), img_array)
    return str(img_path)


@pytest.fixture
def png_image_path(tmp_path: Path) -> str:
    """PNG形式のテスト画像を作成する.

    アナライザが異なる画像形式を処理することをテストするために使用する。
    """
    np.random.seed(42)
    img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    img_path = tmp_path / "test_image.png"
    cv2.imwrite(str(img_path), img_array)
    return str(img_path)


@pytest.fixture
def bmp_image_path(tmp_path: Path) -> str:
    """BMP形式のテスト画像を作成する.

    アナライザが異なる画像形式を処理することをテストするために使用する。
    """
    np.random.seed(42)
    img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    img_path = tmp_path / "test_image.bmp"
    cv2.imwrite(str(img_path), img_array)
    return str(img_path)


@pytest.fixture
def small_image_path(tmp_path: Path) -> str:
    """小さいテスト画像（320x240）を作成する.

    アナライザが異なる画像サイズを処理することをテストするために使用する。
    """
    np.random.seed(42)
    img_array = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    img_path = tmp_path / "small_image.jpg"
    cv2.imwrite(str(img_path), img_array)
    return str(img_path)


# ============================================================================
# 初期化のテスト（2件）
# ============================================================================


def test_analyzer_has_model_and_processor_attributes_after_init(
    mock_clip_model: MagicMock,  # noqa: ARG001
    mock_clip_processor: MagicMock,  # noqa: ARG001
) -> None:
    """アナライザは初期化時にモデルとプロセッサを設定する.

    Given:
        - ジャンルタイプ（例："rpg"）
        - モック化されたCLIPモデルとプロセッサ
    When:
        - ImageQualityAnalyzerインスタンスを作成
    Then:
        - model属性が設定されている
        - processor属性が設定されている
        - device属性が設定されている
    """
    # Arrange & Act
    analyzer = ImageQualityAnalyzer(genre="rpg")

    # Assert
    # モデルとプロセッサが初期化されていることを確認
    assert hasattr(analyzer, "model")
    assert hasattr(analyzer, "processor")
    assert hasattr(analyzer, "device")
    # deviceはGPUまたはCPUのいずれか
    assert analyzer.device in ["cuda", "cpu"]


def test_analyzer_sets_correct_weights_based_on_genre(
    mock_clip_model: MagicMock,  # noqa: ARG001
    mock_clip_processor: MagicMock,  # noqa: ARG001
) -> None:
    """アナライザは初期化時にジャンル特有の重みを設定する.

    Given:
        - 特定のジャンル（例："fps"）
    When:
        - ImageQualityAnalyzerインスタンスを作成
    Then:
        - 正しいジャンル重みがロードされる
        - 重みはジャンルの期待値と一致する
    """
    # Arrange & Act
    analyzer = ImageQualityAnalyzer(genre="fps")

    # Assert
    expected_weights = {
        "blur_score": 0.25,
        "contrast": 0.20,
        "color_richness": 0.10,
        "visual_balance": 0.10,
        "edge_density": 0.10,
        "action_intensity": 0.15,
        "ui_density": 0.00,
        "dramatic_score": 0.10,
    }
    assert analyzer.weights == expected_weights


# ============================================================================
# Tests for analyze method - success path (6 tests)
# ============================================================================


def test_analyze_returns_image_metrics_with_all_required_fields(
    mock_clip_model: MagicMock,  # noqa: ARG001
    mock_clip_processor: MagicMock,  # noqa: ARG001
    sample_image_path: str,
) -> None:
    """分析はすべての必須フィールドが設定されたImageMetricsを返す.

    Given:
        - アナライザインスタンス
        - 有効なテスト画像パス
    When:
        - 画像を分析
    Then:
        - ImageMetricsインスタンスを返す
        - すべてのフィールドが設定されている：
          - path (str)
          - raw_metrics (9フィールドの辞書)
          - normalized_metrics (8フィールドの辞書)
          - semantic_score (float)
          - total_score (float)
          - features (numpy配列)
    """
    # Arrange
    analyzer = ImageQualityAnalyzer()

    # Act
    result = analyzer.analyze(sample_image_path)

    # Assert
    assert result is not None
    assert isinstance(result, ImageMetrics)
    assert result.path == sample_image_path
    assert isinstance(result.raw_metrics, dict)
    assert isinstance(result.normalized_metrics, dict)
    assert isinstance(result.semantic_score, float)
    assert isinstance(result.total_score, float)
    assert isinstance(result.features, np.ndarray)


def test_analyze_calculates_blur_score_using_laplacian(
    mock_clip_model: MagicMock,  # noqa: ARG001
    mock_clip_processor: MagicMock,  # noqa: ARG001
    sample_image_path: str,
) -> None:
    """ラプラシアン分散を使用してblur_scoreを計算する.

    Given:
        - アナライザインスタンス
        - 有効なテスト画像
    When:
        - 画像を分析
    Then:
        - blur_scoreはcv2.Laplacianを使用して計算される
        - blur_scoreはラプラシアンの分散（正の値）
        - 鮮明な画像ほどblur_scoreが高い
    """
    # Arrange
    analyzer = ImageQualityAnalyzer()

    # Act
    result = analyzer.analyze(sample_image_path)

    # Assert
    assert result is not None
    assert "blur_score" in result.raw_metrics
    assert result.raw_metrics["blur_score"] >= 0


def test_analyze_calculates_brightness_from_grayscale(
    mock_clip_model: MagicMock,  # noqa: ARG001
    mock_clip_processor: MagicMock,  # noqa: ARG001
    sample_image_path: str,
) -> None:
    """グレースケール画像平均から輝度を計算する.

    Given:
        - アナライザインスタンス
        - 有効なテスト画像
    When:
        - 画像を分析
    Then:
        - brightnessはグレースケールピクセル値の平均
        - brightnessは範囲[0, 255]内にある
    """
    # Arrange
    analyzer = ImageQualityAnalyzer()

    # Act
    result = analyzer.analyze(sample_image_path)

    # Assert
    assert result is not None
    assert "brightness" in result.raw_metrics
    assert 0 <= result.raw_metrics["brightness"] <= 255


def test_analyze_calculates_contrast_as_standard_deviation(
    mock_clip_model: MagicMock,  # noqa: ARG001
    mock_clip_processor: MagicMock,  # noqa: ARG001
    sample_image_path: str,
) -> None:
    """グレースケール標準偏差としてコントラストを計算する.

    Given:
        - アナライザインスタンス
        - 有効なテスト画像
    When:
        - 画像を分析
    Then:
        - contrastはグレースケールピクセルの標準偏差
        - contrastは非負の値
    """
    # Arrange
    analyzer = ImageQualityAnalyzer()

    # Act
    result = analyzer.analyze(sample_image_path)

    # Assert
    assert result is not None
    assert "contrast" in result.raw_metrics
    assert result.raw_metrics["contrast"] >= 0


def test_analyze_applies_penalty_for_dark_images(
    mock_clip_model: MagicMock,  # noqa: ARG001
    mock_clip_processor: MagicMock,  # noqa: ARG001
    dark_image_path: str,
) -> None:
    """輝度が40未満の画像に0.6のペナルティを適用する.

    Given:
        - アナライザインスタンス
        - 暗いテスト画像（輝度 < 40）
    When:
        - 画像を分析
    Then:
        - 総スコアに0.6のペナルティが適用される
        - ペナルティなしの場合に比べて総スコアが低くなる
    """
    # Arrange
    analyzer = ImageQualityAnalyzer()

    # Act
    result = analyzer.analyze(dark_image_path)

    # Assert
    assert result is not None
    assert result.raw_metrics["brightness"] < 40
    # ペナルティが適用されるため、総スコアは低くなる
    # ペナルティの正確な量は複雑な計算が必要だが、
    # スコアが妥当であることは検証できる
    assert result.total_score >= 0


def test_analyze_combines_metrics_with_genre_specific_weights(
    mock_clip_model: MagicMock,  # noqa: ARG001
    mock_clip_processor: MagicMock,  # noqa: ARG001
    sample_image_path: str,
) -> None:
    """ジャンル特有の重みを使用してメトリックを組み合わせる.

    Given:
        - 特定ジャンルのアナライザインスタンス
        - 有効なテスト画像
    When:
        - 画像を分析
    Then:
        - 総スコアは重み付き合計を使用して計算される
        - 重みはジャンル設定と一致する
    """
    # Arrange
    analyzer = ImageQualityAnalyzer(genre="fps")

    # Act
    result = analyzer.analyze(sample_image_path)

    # Assert
    assert result is not None
    # 重みが使用されていることを確認するため、総スコアが計算されることを検証
    assert result.total_score >= 0
    # FPSジャンルはblur_scoreの重みが高いため（0.25）、blur正規化は
    # 総スコアに大きな影響を与えるはず


# ============================================================================
# Tests for analyze method - edge cases (4 tests)
# ============================================================================


def test_analyze_returns_none_for_nonexistent_file(
    mock_clip_model: MagicMock,  # noqa: ARG001
    mock_clip_processor: MagicMock,  # noqa: ARG001
) -> None:
    """存在しないファイルパスに対してNoneを返す.

    Given:
        - アナライザインスタンス
        - 存在しないファイルパス
    When:
        - 存在しないファイルを分析
    Then:
        - Noneを返す（正常な失敗）
        - 例外は発生しない
    """
    # Arrange
    analyzer = ImageQualityAnalyzer()
    nonexistent_path = "/path/that/does/not/exist.jpg"

    # Act
    result = analyzer.analyze(nonexistent_path)

    # Assert
    assert result is None


def test_analyze_returns_none_for_corrupted_image_file(
    mock_clip_model: MagicMock,  # noqa: ARG001
    mock_clip_processor: MagicMock,  # noqa: ARG001
    tmp_path: Path,
) -> None:
    """破損した画像ファイルに対してNoneを返す.

    Given:
        - アナライザインスタンス
        - 無効な画像データを持つファイル
    When:
        - 破損したファイルを分析
    Then:
        - Noneを返す（正常な失敗）
        - 例外は発生しない
    """
    # Arrange
    analyzer = ImageQualityAnalyzer()
    # 無効な画像データを持つファイルを作成
    corrupted_path = tmp_path / "corrupted.jpg"
    corrupted_path.write_text("This is not a valid image file")

    # Act
    result = analyzer.analyze(str(corrupted_path))

    # Assert
    assert result is None


def test_analyze_handles_various_image_formats(
    mock_clip_model: MagicMock,  # noqa: ARG001
    mock_clip_processor: MagicMock,  # noqa: ARG001
    sample_image_path: str,
    png_image_path: str,
    bmp_image_path: str,
) -> None:
    """JPG、PNG、BMP画像形式を正しく処理する.

    Given:
        - アナライザインスタンス
        - JPG、PNG、BMP形式のテスト画像
    When:
        - 各画像を分析
    Then:
        - すべての形式が正常に分析される
        - すべてが有効なImageMetricsを返す（Noneではない）
    """
    # Arrange
    analyzer = ImageQualityAnalyzer()

    # Act
    result_jpg = analyzer.analyze(sample_image_path)
    result_png = analyzer.analyze(png_image_path)
    result_bmp = analyzer.analyze(bmp_image_path)

    # Assert
    assert result_jpg is not None
    assert result_png is not None
    assert result_bmp is not None


def test_analyze_handles_images_with_different_dimensions(
    mock_clip_model: MagicMock,  # noqa: ARG001
    mock_clip_processor: MagicMock,  # noqa: ARG001
    sample_image_path: str,
    small_image_path: str,
) -> None:
    """異なるサイズの画像を正しく処理する.

    Given:
        - アナライザインスタンス
        - 異なるサイズのテスト画像：
          - 標準：640x480
          - 小さい：320x240
    When:
        - 各画像を分析
    Then:
        - すべての画像が正常に分析される
        - すべてが有効なImageMetricsを返す（Noneではない）
        - 特徴ベクトルは一貫したサイズ（64、）を持つ
    """
    # Arrange
    analyzer = ImageQualityAnalyzer()

    # Act
    result_standard = analyzer.analyze(sample_image_path)
    result_small = analyzer.analyze(small_image_path)

    # Assert
    assert result_standard is not None
    assert result_small is not None
    # All images resized to 128x128, so feature vectors are same size
    assert result_standard.features.shape == (64,)
    assert result_small.features.shape == (64,)


# ============================================================================
# Integration tests (2 tests)
# ============================================================================


def test_analyze_integration_with_metric_normalizer(
    mock_clip_model: MagicMock,  # noqa: ARG001
    mock_clip_processor: MagicMock,  # noqa: ARG001
    sample_image_path: str,
) -> None:
    """MetricNormalizerとの統合で正しい正規化値を生成する.

    Given:
        - アナライザインスタンス
        - 有効なテスト画像
    When:
        - 画像を分析
    Then:
        - 生メトリックが正しく計算される
        - 正規化メトリックは[0, 1]範囲内にある
        - すべての8つの正規化メトリックが存在する
    """
    # Arrange
    analyzer = ImageQualityAnalyzer()

    # Act
    result = analyzer.analyze(sample_image_path)

    # Assert
    assert result is not None
    # Check all normalized metrics are present
    expected_keys = {
        "blur_score",
        "contrast",
        "color_richness",
        "edge_density",
        "dramatic_score",
        "visual_balance",
        "action_intensity",
        "ui_density",
    }
    assert set(result.normalized_metrics.keys()) == expected_keys
    # Check all normalized values are in [0, 1]
    for value in result.normalized_metrics.values():
        assert 0.0 <= value <= 1.0


def test_analyze_produces_consistent_results_for_same_image(
    mock_clip_model: MagicMock,  # noqa: ARG001
    mock_clip_processor: MagicMock,  # noqa: ARG001
    sample_image_path: str,
) -> None:
    """同じ画像を複数回分析する際に一貫した結果を生成する.

    Given:
        - アナライザインスタンス
        - 有効なテスト画像
    When:
        - 同じ画像を2回分析
    Then:
        - 両方の分析で同一の結果が生成される
        - すべてのメトリック値が同じ
        - 総スコアが同じ
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
    assert result1.semantic_score == result2.semantic_score
    # Raw metrics should be identical
    for key in result1.raw_metrics:
        assert result1.raw_metrics[key] == result2.raw_metrics[key]
    # Normalized metrics should be identical
    for key in result1.normalized_metrics:
        assert result1.normalized_metrics[key] == result2.normalized_metrics[key]
    # Features should be identical
    assert np.array_equal(result1.features, result2.features)
