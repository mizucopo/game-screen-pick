"""ImageQualityAnalyzerPoolの単体テスト.

このテストモジュールは以下のベストプラクティスに従っています：
1. 「How」（実装詳細）ではなく「What」（観察可能な挙動）をテスト
2. CLIPモデルのみ戦略的にモック化（700MB、10-30秒のロード時間）
3. 明確なコメント付きのAAAパターン（Arrange, Act, Assert）を使用
4. 高速実行（約2-5秒） - 重いモデルロードなし
"""

from typing import Any

import pytest

from src.analyzers.image_quality_analyzer_pool import ImageQualityAnalyzerPool
from src.models.image_metrics import ImageMetrics


class _FakePool:
    """multiprocessing.Pool互換のテスト用フェイク.

    実プロセスを生成せずに同等の呼び出しフローを再現し、
    CI環境でのfork起因ハングを回避する。
    """

    def __init__(
        self,
        processes: int | None = None,  # noqa: ARG002
        initializer: Any | None = None,
        initargs: tuple[Any, ...] = (),
    ) -> None:
        if initializer is not None:
            initializer(*initargs)

    def map(self, func: Any, items: list[str]) -> list[ImageMetrics | None]:
        return [func(item) for item in items]

    def close(self) -> None:
        return

    def join(self) -> None:
        return


@pytest.fixture(autouse=True)
def use_fake_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """ImageQualityAnalyzerPoolが使うPool実装をフェイクに差し替える."""
    monkeypatch.setattr("src.analyzers.image_quality_analyzer_pool.Pool", _FakePool)


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
