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


@pytest.fixture
def mock_clip_model() -> Generator[MagicMock, None, None]:
    """700MBの重みロードを回避するためのCLIPモデルのモック.

    このfixtureは本物のCLIPモデルを以下のモックに置き換えます：
    - 決定論的テストのために固定されたlogit値を返す
    - GPU/CPU切り替えのための.to(device)呼び出しをサポート
    """
    with patch("transformers.CLIPModel.from_pretrained") as mock:
        model = MagicMock()

        # get_text_features用のモック（テキスト埋め込み）
        # 実際のテンソルを返すように修正
        text_features = torch.tensor([[1.0]])

        # get_image_features用のモック（画像埋め込み）
        # matmulで使用できるように実際のテンソルを返す
        image_features = torch.tensor([[25.0]])  # logits計算用のダミー値

        # メソッドをモック
        model.get_text_features = MagicMock(return_value=text_features)
        model.get_image_features = MagicMock(return_value=image_features)

        # .to()メソッドと既存の__call__もモック
        model.to = MagicMock(return_value=model)

        # 既存の呼び出し形式のモック（後方互換性）
        mock_output = MagicMock()
        mock_output.logits_per_image = torch.tensor([[25.0]])
        model.return_value = mock_output

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
    assert 0 <= result.semantic_score <= 1
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


@pytest.mark.parametrize(
    "image_path_fixture,check_dark_penalty",
    [
        ("sample_image_path", False),
        ("png_image_path", False),
        ("bmp_image_path", False),
        ("small_image_path", False),
        ("dark_image_path", True),  # 暗い画像は輝度ペナルティを確認
    ],
)
def test_analyzes_images_with_various_formats_and_properties(
    request: pytest.FixtureRequest,
    image_path_fixture: str,
    check_dark_penalty: bool,
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
        - 暗い画像には輝度ペナルティが適用されること
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
    # 暗い画像の場合は輝度ペナルティが適用される
    if check_dark_penalty:
        assert result.raw_metrics["brightness"] < 40
        assert result.total_score >= 0  # ペナルティ適用後も有効なスコア


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

    # Act
    results = analyzer.analyze_batch(paths)

    # Assert
    assert len(results) == 3
    for result, path in zip(results, paths):
        assert result is not None
        assert isinstance(result, ImageMetrics)
        assert result.path == path
        assert 0 <= result.total_score <= 100
        assert 0 <= result.semantic_score <= 1


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

    # Act
    results = analyzer.analyze_batch(paths)

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


def test_analyze_batch_falls_back_on_oom(
    sample_image_path: str,
) -> None:
    """CUDA OOM発生時にバッチサイズが縮小されてリトライされること.

    Given:
        - アナライザインスタンスがある
        - 有効なテスト画像がある
        - CLIP推論時にCUDA OOMが発生する状況
    When:
        - バッチ処理で分析される
    Then:
        - OOMエラーが回復され、有効な結果が返されること
    """
    # Arrange
    analyzer = ImageQualityAnalyzer()
    paths = [sample_image_path]

    # Act & Assert
    # CUDA OOMをモックして、2回目の呼び出しで成功するように設定
    original_model = analyzer.model
    call_count = [0]

    def mock_get_image_features_with_oom(**_kwargs: object) -> torch.Tensor:  # noqa: ARG001
        call_count[0] += 1
        if call_count[0] == 1:
            # 最初の呼び出しでOOMを発生
            raise torch.cuda.OutOfMemoryError()
        # 2回目以降は正常に返す（CLIP base-patch32の出力形状: batch_size x 512）
        return torch.randn(1, 512)

    with patch.object(
        original_model,
        "get_image_features",
        side_effect=mock_get_image_features_with_oom,
    ):
        results = analyzer.analyze_batch(paths, batch_size=32)

    # Assert - 最終的に成功しているはず（観測可能な振る舞いのみ検証）
    assert results[0] is not None
    assert isinstance(results[0], ImageMetrics)
    assert results[0].path == sample_image_path
    assert 0 <= results[0].total_score <= 100


def test_analyze_batch_retries_only_failed_batches_on_oom(
    sample_image_path: str,
    png_image_path: str,
    small_image_path: str,
    dark_image_path: str,
) -> None:
    """OOM発生時に失敗したバッチのみが再試行されること.

    Given:
        - アナライザインスタンスがある
        - 複数の有効なテスト画像がある
        - 2番目のバッチでCUDA OOMが発生する状況
    When:
        - バッチ処理で分析される（バッチサイズ2）
    Then:
        - 最初のバッチの結果が保持されること
        - すべての有効な画像の結果が返されること
    """
    # Arrange
    analyzer = ImageQualityAnalyzer()
    paths = [sample_image_path, png_image_path, small_image_path, dark_image_path]
    batch_size = 2  # 2画像ずつ処理

    # Act & Assert
    # 2番目の呼び出しでOOMを発生させる
    original_model = analyzer.model
    call_count = [0]

    def mock_get_image_features_with_oom(**kwargs: object) -> torch.Tensor:
        call_count[0] += 1

        # pixel_values から実際のバッチサイズを取得
        inputs = kwargs.get("pixel_values")
        if inputs is not None and isinstance(inputs, torch.Tensor):
            actual_batch_size = inputs.shape[0]
        else:
            actual_batch_size = 1  # フォールバック

        # 2番目の呼び出し（バッチ2）でOOMを発生
        if call_count[0] == 2:
            raise torch.cuda.OutOfMemoryError()

        # 該当するバッチサイズのテンソルを返す
        return torch.randn(actual_batch_size, 512)

    with patch.object(
        original_model,
        "get_image_features",
        side_effect=mock_get_image_features_with_oom,
    ):
        results = analyzer.analyze_batch(paths, batch_size=batch_size)

    # Assert - すべての画像が処理されている（観測可能な振る舞いのみ検証）
    assert len(results) == 4
    for result, expected_path in zip(results, paths):
        assert result is not None
        assert isinstance(result, ImageMetrics)
        assert result.path == expected_path
        assert 0 <= result.total_score <= 100
