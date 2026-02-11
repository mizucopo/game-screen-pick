"""ImageQualityAnalyzerPoolの単体テスト.

このテストモジュールは以下のベストプラクティスに従っています：
1. 「How」（実装詳細）ではなく「What」（観察可能な挙動）をテスト
2. CLIPモデルのみ戦略的にモック化（700MB、10-30秒のロード時間）
3. 明確なコメント付きのAAAパターン（Arrange, Act, Assert）を使用
4. 高速実行（約2-5秒） - 重いモデルロードなし
"""

from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import torch

from src.analyzers.image_quality_analyzer_pool import ImageQualityAnalyzerPool
from src.models.image_metrics import ImageMetrics


@pytest.fixture(autouse=True)
def mock_clip_model() -> Generator[Any, Any, Any]:
    """700MBの重みロードを回避するためのCLIPモデルのモック.

    512次元の正規化されたCLIP特徴ベクトルを返す（実際のモデルと同じ形状）.
    バッチサイズに応じた形状を動的に返す.
    """
    with patch("transformers.CLIPModel.from_pretrained") as mock:
        model = MagicMock()

        # get_text_features用のモック（テキスト埋め込み）: (batch_size, 512)
        def mock_get_text_features(**kwargs: object) -> torch.Tensor:
            inputs = kwargs.get("input_ids")
            if inputs is not None and isinstance(inputs, torch.Tensor):
                batch_size = inputs.shape[0]
            else:
                batch_size = 1
            return torch.ones(batch_size, 512) / torch.sqrt(torch.tensor(512.0))

        # get_image_features用のモック（画像埋め込み）: (batch_size, 512)
        def mock_get_image_features(**kwargs: object) -> torch.Tensor:
            inputs = kwargs.get("pixel_values")
            if inputs is not None and isinstance(inputs, torch.Tensor):
                batch_size = inputs.shape[0]
            else:
                batch_size = 1
            return torch.ones(batch_size, 512) / torch.sqrt(torch.tensor(512.0))

        model.get_text_features = MagicMock(side_effect=mock_get_text_features)
        model.get_image_features = MagicMock(side_effect=mock_get_image_features)
        model.to = MagicMock(return_value=model)
        model.eval = MagicMock()

        mock.return_value = model
        yield mock


@pytest.fixture(autouse=True)
def mock_clip_processor() -> Generator[Any, Any, Any]:
    """トークナイザと特徴抽出器のロードを回避するためのCLIPプロセッサのモック.

    呼び出し時に辞書のようなオブジェクトを返し、.to()メソッドをサポート.
    バッチサイズに応じた形状を動的に返す. 固定値を使用して決定論的にする.
    """
    with patch("transformers.CLIPProcessor.from_pretrained") as mock:
        processor = MagicMock()

        def mock_processor(**kwargs: object) -> MagicMock:
            images = kwargs.get("images")
            if images is not None:
                if isinstance(images, list):
                    batch_size = len(images)
                else:
                    batch_size = 1
            else:
                batch_size = 1

            input_ids = torch.tensor([[1, 2, 3]])
            pixel_values = torch.ones(batch_size, 3, 224, 224) * 0.5
            attention_mask = torch.tensor([[1, 1, 1]])

            result_obj = MagicMock()
            result_obj.input_ids = input_ids
            result_obj.pixel_values = pixel_values
            result_obj.attention_mask = attention_mask

            # .to()メソッドをサポート（呼ばれたときも同じ属性を持つオブジェクトを返す）
            def to_method(_device: str) -> MagicMock:
                to_result = MagicMock()
                to_result.input_ids = input_ids
                to_result.pixel_values = pixel_values
                to_result.attention_mask = attention_mask
                to_result.to = MagicMock(side_effect=to_method)
                return to_result

            result_obj.to = MagicMock(side_effect=to_method)

            return result_obj

        processor.side_effect = mock_processor

        mock.return_value = processor
        yield mock


@pytest.fixture
def sample_image_path(tmp_path: Path) -> str:
    """標準的なテスト画像（640x480 JPG）を作成する."""
    np.random.seed(42)
    img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    img_path = tmp_path / "test_image.jpg"
    cv2.imwrite(str(img_path), img_array)
    return str(img_path)


@pytest.fixture
def multiple_image_paths(tmp_path: Path) -> list[str]:
    """複数のテスト画像を作成する."""
    paths = []
    for i in range(3):
        np.random.seed(42 + i)
        img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img_path = tmp_path / f"test_image_{i}.jpg"
        cv2.imwrite(str(img_path), img_array)
        paths.append(str(img_path))
    return paths


def test_pool_context_manager_starts_and_closes_pool() -> None:
    """プールがコンテキストマネージャーで正常に開始・終了すること.

    Given:
        - ImageQualityAnalyzerPoolインスタンスがある
    When:
        - コンテキストマネージャーとして使用する
    Then:
        - プールが開始されること
        - コンテキスト終了時にプールが閉じられること
    """
    # Arrange
    pool = ImageQualityAnalyzerPool(genre="mixed", num_workers=2, force_cpu=True)

    # Act & Assert
    with pool:
        # コンテキスト内ではプールが開始されている
        assert pool.num_workers == 2

    # コンテキスト終了後はプールが閉じられている
    with pytest.raises(RuntimeError, match="Pool is not started"):
        _ = pool.num_workers


def test_pool_analyze_batch_processes_multiple_images(
    multiple_image_paths: list[str],
) -> None:
    """プールが複数の画像を並列処理すること.

    Given:
        - モックされたCLIPモデルがある
        - 開始されたプールがある
        - 複数の有効なテスト画像がある
    When:
        - analyze_batchが呼び出される
    Then:
        - すべての画像に対して有効なImageMetricsが返されること
        - 結果の数が入力数と一致すること
    """
    # Arrange
    with ImageQualityAnalyzerPool(genre="mixed", num_workers=2, force_cpu=True) as pool:
        # Act
        results = pool.analyze_batch(multiple_image_paths)

        # Assert
        assert len(results) == 3
        for result, path in zip(results, multiple_image_paths):
            assert result is not None
            assert isinstance(result, ImageMetrics)
            assert result.path == path
            assert 0 <= result.total_score <= 100


def test_pool_analyze_batch_handles_mixed_valid_and_invalid_images(
    sample_image_path: str,
) -> None:
    """プールが有効な画像と無効な画像が混在する場合に正しく処理すること.

    Given:
        - モックされたCLIPモデルがある
        - 開始されたプールがある
        - 有効な画像パスと存在しないパスが混在している
    When:
        - analyze_batchが呼び出される
    Then:
        - 有効な画像にはImageMetricsが返されること
        - 無効なパスにはNoneが返されること
        - 結果の数が入力数と一致すること
    """
    # Arrange
    nonexistent_path = "/path/that/does/not/exist.jpg"
    paths = [sample_image_path, nonexistent_path, sample_image_path]

    with ImageQualityAnalyzerPool(genre="mixed", num_workers=2, force_cpu=True) as pool:
        # Act
        results = pool.analyze_batch(paths)

        # Assert
        assert len(results) == 3
        assert results[0] is not None
        assert results[1] is None  # 存在しないパス
        assert results[2] is not None


def test_pool_raises_error_when_not_started() -> None:
    """プールが開始されていない状態でanalyze_batchを呼び出すとエラーが発生すること.

    Given:
        - 開始されていないプールがある
    When:
        - analyze_batchが呼び出される
    Then:
        - RuntimeErrorが発生すること
    """
    # Arrange
    pool = ImageQualityAnalyzerPool(genre="mixed", num_workers=2, force_cpu=True)

    # Act & Assert
    with pytest.raises(RuntimeError, match="Pool is not started"):
        pool.analyze_batch(["dummy.jpg"])
