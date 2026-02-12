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


def test_pool_context_manager_starts_and_closes_pool() -> None:
    """プールがコンテキストマネージャーで正常に開始・終了すること.

    Given:
        - ImageQualityAnalyzerPoolインスタンスがある
    When:
        - コンテキストマネージャーとして使用する
    Then:
        - コンテキスト内で正常に動作すること
        - コンテキスト終了後にプールが閉じられること
    """
    # Arrange
    pool = ImageQualityAnalyzerPool(genre="mixed", num_workers=2, force_cpu=True)

    # Act & Assert
    with pool:
        # コンテキスト内で正常に動作
        pass

    # コンテキスト終了後はプールが閉じられている
    with pytest.raises(RuntimeError, match="Pool is not started"):
        pool.analyze_batch(["dummy.jpg"])


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
