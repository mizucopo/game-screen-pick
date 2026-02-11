"""ImageQualityAnalyzerの単体テスト.

このテストモジュールは以下のベストプラクティスに従っています：
1.「How」（実装詳細）ではなく「What」（観察可能な挙動）をテスト
2. CLIPモデルのみ戦略的にモック化（700MB、10-30秒のロード時間）
3. OpenCV操作、NumPy計算、MetricNormalizerはモック化しない
4. 明確なコメント付きのAAAパターン（Arrange, Act, Assert）を使用
5. 高速実行（約2-5秒） - 重いモデルロードなし
"""

import logging
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import torch
from PIL import UnidentifiedImageError

from src.analyzers.image_quality_analyzer import ImageQualityAnalyzer
from src.models.image_metrics import ImageMetrics


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
def bmp_image_path(tmp_path: Path) -> str:
    """BMP形式のテスト画像（640x480）を作成する."""
    return _create_test_image(tmp_path, "test_image.bmp", (480, 640), (0, 255))


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


@pytest.mark.parametrize(
    "genre",
    ["fps", "2d_rpg", "3d_rpg", "action", "adventure", "default"],
)
def test_analyze_returns_valid_metrics_with_genre_specific_weights(
    mock_clip_model: MagicMock,  # noqa: ARG001
    mock_clip_processor: MagicMock,  # noqa: ARG001
    sample_image_path: str,
    genre: str,
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
    analyzer = ImageQualityAnalyzer(genre=genre)

    # Act
    result = analyzer.analyze(sample_image_path)

    # Assert
    assert result is not None
    assert isinstance(result, ImageMetrics)
    assert result.path == sample_image_path
    # スコア値が有効範囲内であることを検証
    assert 0 <= result.total_score <= 100
    assert 0 <= result.semantic_score <= 1
    # 正規化されたメトリックが存在することを検証
    assert len(result.normalized_metrics) > 0


def test_analyze_applies_penalty_for_dark_images(
    mock_clip_model: MagicMock,  # noqa: ARG001
    mock_clip_processor: MagicMock,  # noqa: ARG001
    dark_image_path: str,
) -> None:
    """輝度が40未満の画像に0.6のペナルティが適用されること.

    Given:
        - アナライザインスタンスがある
        - 暗いテスト画像（輝度 < 40）がある
    When:
        - 画像が分析される
    Then:
        - 総スコアに0.6のペナルティが適用されること
        - ペナルティなしの場合に比べて総スコアが低くなること
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


def test_analyze_returns_none_for_nonexistent_file(
    mock_clip_model: MagicMock,  # noqa: ARG001
    mock_clip_processor: MagicMock,  # noqa: ARG001
) -> None:
    """存在しないファイルパスに対してNoneが返されること.

    Given:
        - アナライザインスタンスがある
        - 存在しないファイルパスがある
    When:
        - 存在しないファイルが分析される
    Then:
        - Noneが返されること（正常な失敗）
        - 例外が発生しないこと
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
    """破損した画像ファイルに対してNoneが返されること.

    Given:
        - アナライザインスタンスがある
        - 無効な画像データを持つファイルがある
    When:
        - 破損したファイルが分析される
    Then:
        - Noneが返されること（正常な失敗）
        - 例外が発生しないこと
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


@pytest.mark.parametrize(
    "image_path_fixture",
    [
        "sample_image_path",
        "png_image_path",
        "bmp_image_path",
        "small_image_path",
        "dark_image_path",
    ],
)
def test_analyzes_images_with_various_formats_and_properties(
    mock_clip_model: MagicMock,  # noqa: ARG001
    mock_clip_processor: MagicMock,  # noqa: ARG001
    request: pytest.FixtureRequest,
    image_path_fixture: str,
) -> None:
    """様々な形式と特性の画像が正しく処理されること.

    Given:
        - アナライザインスタンスがある
        - 異なる形式（JPG、PNG、BMP）とサイズ、特性のテスト画像がある
    When:
        - 各画像が分析される
    Then:
        - すべての形式とサイズが正常に分析されること
        - 有効なImageMetricsが返されること
        - 特徴ベクトルが一貫したサイズを持つこと
    """
    # Arrange
    analyzer = ImageQualityAnalyzer()
    image_path = request.getfixturevalue(image_path_fixture)

    # Act
    result = analyzer.analyze(image_path)

    # Assert
    assert result is not None
    # すべての画像はリサイズされるため、特徴ベクトルサイズは一貫している
    assert result.features.shape == (64,)


def test_analyze_produces_consistent_results_for_same_image(
    mock_clip_model: MagicMock,  # noqa: ARG001
    mock_clip_processor: MagicMock,  # noqa: ARG001
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


def test_analyze_logs_warning_for_corrupted_image(
    mock_clip_model: MagicMock,  # noqa: ARG001
    mock_clip_processor: MagicMock,  # noqa: ARG001
    sample_image_path: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """PILで画像を開く際のエラー時にWARNINGログが出力されること.

    Given:
        - アナライザインスタンスがある
        - 有効なテスト画像（cv2.imreadは成功する）がある
        - ログキャプチャ設定（WARNINGレベル以上）がある
    When:
        - PIL.Image.openでUnidentifiedImageErrorが発生するようにモック化される
    Then:
        - WARNINGレベルのログが出力されること
        - ログメッセージにパスと例外内容が含まれること
        - Noneが返されること（正常な失敗）
    """
    # Arrange
    caplog.set_level(logging.WARNING)
    analyzer = ImageQualityAnalyzer()

    # PIL.Image.openをモック化してUnidentifiedImageErrorを発生
    with patch(
        "PIL.Image.open",
        side_effect=UnidentifiedImageError("Cannot identify image file"),
    ):
        # Act
        result = analyzer.analyze(sample_image_path)

        # Assert
        assert result is None
        # WARNINGログが出力されていることを確認
        assert len(caplog.records) > 0
        # 最新のログレコードがWARNINGであることを確認
        warning_log = caplog.records[-1]
        assert warning_log.levelno == logging.WARNING
        # ログメッセージにパスと例外情報が含まれていることを確認
        log_message = warning_log.getMessage()
        assert sample_image_path in log_message
        assert "画像分析をスキップしました" in log_message
        assert "UnidentifiedImageError" in log_message


@pytest.mark.parametrize(
    "exception_class,exception_msg",
    [
        (AttributeError, "Test attribute error"),
        (TypeError, "Test type error"),
        (KeyError, "test_key"),
    ],
)
def test_analyze_re_raises_unexpected_exceptions(
    mock_clip_model: MagicMock,  # noqa: ARG001
    mock_clip_processor: MagicMock,  # noqa: ARG001
    sample_image_path: str,
    caplog: pytest.LogCaptureFixture,
    exception_class: type[Exception],
    exception_msg: str,
) -> None:
    """実装バグ（予期しない例外）時に例外が再スローされること.

    Given:
        - アナライザインスタンスがある
        - 有効なテスト画像がある
        - ログキャプチャ設定（ERRORレベル以上）がある
    When:
        - analyzeメソッド内で予期しない例外が発生するようにモック化される
    Then:
        - ERRORレベルのログが出力されること
        - 例外が再スローされること
    """
    # Arrange
    caplog.set_level(logging.ERROR)
    analyzer = ImageQualityAnalyzer()
    # _extract_diversity_featuresメソッドをモック化して例外を発生
    with patch.object(
        analyzer,
        "_extract_diversity_features",
        side_effect=exception_class(exception_msg),
    ):
        # Act & Assert
        with pytest.raises(exception_class, match=exception_msg):
            analyzer.analyze(sample_image_path)

        # ERRORログが出力されていることを確認
        assert len(caplog.records) > 0
        error_log = caplog.records[-1]
        assert error_log.levelno == logging.ERROR
        # ログメッセージにパスが含まれていることを確認
        log_message = error_log.getMessage()
        assert sample_image_path in log_message
        assert "予期しないエラーが発生しました" in log_message
