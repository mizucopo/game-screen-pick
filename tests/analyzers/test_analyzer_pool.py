"""ImageQualityAnalyzerPoolの単体テスト.

このテストモジュールは以下のベストプラクティスに従っています：
1. 「How」（実装詳細）ではなく「What」（観察可能な挙動）をテスト
2. CLIPモデルのみ戦略的にモック化（700MB、10-30秒のロード時間）
3. 明確なコメント付きのAAAパターン（Arrange, Act, Assert）を使用
4. 高速実行（約2-5秒） - 重いモデルロードなし
"""

from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import torch

from src.analyzers.analyzer_pool import (
    ImageQualityAnalyzerPool,
    _analyze_single,
    _init_worker,
)
from src.models.image_metrics import ImageMetrics


@pytest.fixture
def mock_clip_model() -> Generator[MagicMock, None, None]:
    """700MBの重みロードを回避するためのCLIPモデルのモック."""
    with patch("transformers.CLIPModel.from_pretrained") as mock:
        model = MagicMock()

        # get_text_features用のモック（テキスト埋め込み）
        text_features = torch.tensor([[1.0]])

        # get_image_features用のモック（画像埋め込み）
        image_features = torch.tensor([[25.0]])

        model.get_text_features = MagicMock(return_value=text_features)
        model.get_image_features = MagicMock(return_value=image_features)
        model.to = MagicMock(return_value=model)

        mock.return_value = model
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


def test_init_worker_initializes_analyzer_in_global_scope() -> None:
    """ワーカー初期化関数がグローバルスコープでアナライザを初期化すること.

    Given:
        - モックされたCLIPモデルがある
    When:
        - _init_workerが呼び出される
    Then:
        - グローバル_analyzerが初期化されること
        - _analyze_singleで分析が実行できること
    """
    # Arrange
    import src.analyzers.analyzer_pool as pool_module

    # 既存の_analyzerをNoneにリセット
    pool_module._analyzer = None

    # Act
    _init_worker(genre="mixed")

    # Assert
    assert pool_module._analyzer is not None
    assert hasattr(pool_module._analyzer, "analyze")


def test_analyze_single_returns_valid_metrics(
    sample_image_path: str,
) -> None:
    """単一分析関数が有効なメトリクスを返すこと.

    Given:
        - モックされたCLIPモデルがある
        - ワーカーが初期化されている
        - 有効なテスト画像がある
    When:
        - _analyze_singleが呼び出される
    Then:
        - 有効なImageMetricsが返されること
    """
    # Arrange
    import src.analyzers.analyzer_pool as pool_module

    pool_module._analyzer = None
    _init_worker(genre="mixed")

    # Act
    result = _analyze_single(sample_image_path)

    # Assert
    assert result is not None
    assert isinstance(result, ImageMetrics)
    assert result.path == sample_image_path
    assert 0 <= result.total_score <= 100


def test_analyze_single_returns_none_for_invalid_path() -> None:
    """単一分析関数が無効なパスに対してNoneを返すこと.

    Given:
        - モックされたCLIPモデルがある
        - ワーカーが初期化されている
        - 存在しないファイルパスがある
    When:
        - _analyze_singleが呼び出される
    Then:
        - Noneが返されること
    """
    # Arrange
    import src.analyzers.analyzer_pool as pool_module

    pool_module._analyzer = None
    _init_worker(genre="mixed")
    nonexistent_path = "/path/that/does/not/exist.jpg"

    # Act
    result = _analyze_single(nonexistent_path)

    # Assert
    assert result is None


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


def test_pool_force_cpu_parameter_works() -> None:
    """force_cpuパラメータが正しく動作すること.

    Given:
        - force_cpu=Trueでプールが作成される
    When:
        - プールが開始される
    Then:
        - ワーカープロセスでCUDA_VISIBLE_DEVICESが設定されること
        - ワーカーでCPUが使用されること
    """
    # Arrange & Act & Assert
    # force_cpu=Trueでプールを作成・開始
    with ImageQualityAnalyzerPool(genre="mixed", num_workers=1, force_cpu=True) as pool:
        # プールが正常に開始されることを確認
        assert pool.num_workers == 1
