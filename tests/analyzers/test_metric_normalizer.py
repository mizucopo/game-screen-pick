"""MetricNormalizerの単体テスト.

このテストモジュールは以下のベストプラクティスに従っています：
1.「How」（実装詳細）ではなく「What」（観察可能な挙動）をテスト
2. モックなし - 予測可能な入出力を持つ純粋な関数テスト
3. 明確なコメント付きのAAAパターン（Arrange, Act, Assert）を使用
4. 高速実行（約0.1秒） - 外部依存関係なし
"""

from src.analyzers.metric_normalizer import MetricNormalizer
from src.models.normalized_metrics import NormalizedMetrics
from src.models.raw_metrics import RawMetrics


def test_sigmoid_with_default_steepness_produces_expected_curve() -> None:
    """デフォルトの急峻さで期待されるシグモイド曲線が生成されること.

    Given:
        - center値が50.0である
        - デフォルトの急峻さが0.1である
        - 3つのテストポイント：centerの下、center、centerの上がある
    When:
        - 各ポイントのsigmoidが計算される
    Then:
        - 期待される値を持つ滑らかな曲線が生成されること
        - 単調増加特性が保持されること
    """
    # Arrange
    center = 50.0
    steepness = 0.1

    # Act
    result_below = MetricNormalizer.sigmoid(0.0, center, steepness)
    result_at = MetricNormalizer.sigmoid(50.0, center, steepness)
    result_above = MetricNormalizer.sigmoid(100.0, center, steepness)

    # Assert
    assert result_below < 0.5
    assert result_at == 0.5
    assert result_above > 0.5
    assert result_below < result_at < result_above


def test_sigmoid_handles_overflow_without_crashing() -> None:
    """例外を発生させずにオーバーフロー/アンダーフローが適切に処理されること.

    Given:
        - 極端な入力値（1e10, -1e10）がある
    When:
        - 極端な値でsigmoidが計算される
    Then:
        - 極端な正の値に対して1.0が返されること
        - 極端な負の値に対して0.0が返されること
    """
    # Arrange
    center = 500.0

    # Act
    result_positive = MetricNormalizer.sigmoid(1e10, center)
    result_negative = MetricNormalizer.sigmoid(-1e10, center)

    # Assert
    assert result_positive == 1.0
    assert result_negative == 0.0


def test_normalize_all_returns_all_expected_metrics() -> None:
    """すべての正規化メトリクスを含むオブジェクトが返されること.

    Given:
        - すべての必須フィールドを含む生メトリクスオブジェクトがある
    When:
        - normalize_allが呼び出される
    Then:
        - 期待されるすべてのメトリクスが返されること
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
