"""MetricNormalizerの単体テスト.

このテストモジュールは以下のベストプラクティスに従っています：
1. 「What」（観察可能な挙動）をテスト
2. モックなし - 予測可能な入出力を持つ純粋な関数テスト
3. 明確なコメント付きのAAAパターン（Arrange, Act, Assert）を使用
"""

from src.analyzers.metric_normalizer import MetricNormalizer
from src.models.normalized_metrics import NormalizedMetrics
from src.models.raw_metrics import RawMetrics


def test_normalize_all_returns_metrics_in_valid_range() -> None:
    """すべての正規化メトリクスが有効範囲[0, 1]で返されること.

    Given:
        - すべての必須フィールドを含む生メトリクスオブジェクトがある
    When:
        - normalize_allが呼び出される
    Then:
        - すべての正規化値が[0, 1]範囲内にあること
    """
    # Arrange
    raw = RawMetrics(
        blur_score=500.0,
        brightness=100.0,
        contrast=50.0,
        color_richness=40.0,
        edge_density=0.2,
        dramatic_score=50.0,
        visual_balance=80.0,
        action_intensity=30.0,
        ui_density=10.0,
    )

    # Act
    result = MetricNormalizer.normalize_all(raw)

    # Assert
    assert isinstance(result, NormalizedMetrics)
    assert 0.0 <= result.blur_score <= 1.0
    assert 0.0 <= result.contrast <= 1.0
    assert 0.0 <= result.color_richness <= 1.0
    assert 0.0 <= result.edge_density <= 1.0
    assert 0.0 <= result.dramatic_score <= 1.0
    assert 0.0 <= result.visual_balance <= 1.0
    assert 0.0 <= result.action_intensity <= 1.0
    assert 0.0 <= result.ui_density <= 1.0
