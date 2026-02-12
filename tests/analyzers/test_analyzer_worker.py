"""AnalyzerWorkerの単体テスト.

このテストモジュールは以下のベストプラクティスに従っています：
1. 「How」（実装詳細）ではなく「What」（観察可能な挙動）をテスト
2. CLIPモデルのみ戦略的にモック化（700MB、10-30秒のロード時間）
3. 明確なコメント付きのAAAパターン（Arrange, Act, Assert）を使用
4. 高速実行（約2-5秒） - 重いモデルロードなし
"""

import pytest

from src.analyzers.analyzer_worker import AnalyzerWorker
from src.models.image_metrics import ImageMetrics


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
