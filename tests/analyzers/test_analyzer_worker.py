"""AnalyzerWorkerの単体テスト.

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

from src.analyzers.analyzer_worker import AnalyzerWorker
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


def test_init_worker_initializes_class_variable() -> None:
    """init_worker staticmethodがクラス変数_workerを初期化すること.

    Given:
        - AnalyzerWorkerクラスがある
    When:
        - init_worker staticmethodを呼び出す
    Then:
        - クラス変数_workerがAnalyzerWorkerインスタンスに設定されること
        - _workerのanalyzer属性が初期化されていること
    """
    # Arrange
    AnalyzerWorker._worker = None  # クリーンな状態

    # Act
    AnalyzerWorker.init_worker(genre="mixed", force_cpu=True)

    # Assert
    assert AnalyzerWorker._worker is not None
    assert AnalyzerWorker._worker.analyzer is not None
    assert AnalyzerWorker._worker.analyzer.device == "cpu"


def test_analyze_single_returns_metrics_for_valid_image(
    sample_image_path: str,
) -> None:
    """analyze_single staticmethodが有効な画像に対してImageMetricsを返すこと.

    Given:
        - モックされたCLIPモデルがある
        - 初期化されたAnalyzerWorkerがある
        - 有効な画像ファイルパスがある
    When:
        - analyze_single staticmethodを呼び出す
    Then:
        - 正常にImageMetricsが返されること
        - パスが正しく設定されていること
        - スコアが有効範囲内であること
    """
    # Arrange
    AnalyzerWorker.init_worker(genre="mixed", force_cpu=True)

    # Act
    result = AnalyzerWorker.analyze_single(sample_image_path)

    # Assert
    assert result is not None
    assert isinstance(result, ImageMetrics)
    assert result.path == sample_image_path
    assert 0 <= result.total_score <= 100


def test_analyze_single_raises_error_when_not_initialized() -> None:
    """analyze_single staticmethodが未初期化時にRuntimeErrorを発生させること.

    Given:
        - AnalyzerWorkerクラスがある
        - クラス変数_workerがNone（未初期化）
    When:
        - analyze_single staticmethodを呼び出す
    Then:
        - RuntimeErrorが発生すること
        - エラーメッセージに適切な内容が含まれること
    """
    # Arrange
    AnalyzerWorker._worker = None  # 未初期化状態

    # Act & Assert
    with pytest.raises(RuntimeError, match="AnalyzerWorker not initialized"):
        AnalyzerWorker.analyze_single("dummy.jpg")


def test_worker_analyze_method_returns_metrics(
    sample_image_path: str,
) -> None:
    """WorkerインスタンスのanalyzeメソッドがImageMetricsを返すこと.

    Given:
        - モックされたCLIPモデルがある
        - 作成されたAnalyzerWorkerインスタンスがある
        - 有効な画像ファイルパスがある
    When:
        - analyzeインスタンスメソッドを呼び出す
    Then:
        - 正常にImageMetricsが返されること
        - 結果の内容が正しいこと
    """
    # Arrange
    worker = AnalyzerWorker(genre="mixed", force_cpu=True)

    # Act
    result = worker.analyze(sample_image_path)

    # Assert
    assert result is not None
    assert isinstance(result, ImageMetrics)
    assert result.path == sample_image_path
    assert 0 <= result.total_score <= 100
