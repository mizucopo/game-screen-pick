"""MetricNormalizerの単体テスト.

このテストモジュールは以下のベストプラクティスに従っています：
1.「How」（実装詳細）ではなく「What」（観察可能な挙動）をテスト
2. モックなし - 予測可能な入出力を持つ純粋な関数テスト
3. 明確なコメント付きのAAAパターン（Arrange, Act, Assert）を使用
4. 高速実行（約0.1秒） - 外部依存関係なし
"""

from src.analyzers.metric_normalizer import MetricNormalizer


def test_sigmoid_with_default_steepness_produces_expected_curve() -> None:
    """デフォルトの急峻さで期待されるsigmoid曲線が生成されること.

    Given:
        - center値が50.0（コントラストメトリックの典型的な値）である
        - デフォルトの急峻さが0.1である
        - 3つのテストポイント：centerの下、center、centerの上がある
    When:
        - 各ポイントのsigmoidが計算される
    Then:
        - 期待される値を持つ滑らかな曲線が生成されること
        - より小さい値はより小さい出力を生成すること
        - より大きい値はより大きい出力を生成すること
    """
    # Arrange
    center = 50.0
    steepness = 0.1
    x_below = 0.0
    x_at = 50.0
    x_above = 100.0

    # Act
    result_below = MetricNormalizer.sigmoid(x_below, center, steepness)
    result_at = MetricNormalizer.sigmoid(x_at, center, steepness)
    result_above = MetricNormalizer.sigmoid(x_above, center, steepness)

    # Assert
    assert result_below < 0.5
    assert result_at == 0.5
    assert result_above > 0.5
    # 単調増加特性の確認
    assert result_below < result_at < result_above


def test_sigmoid_handles_overflow_without_crashing() -> None:
    """例外を発生させずにオーバーフロー/アンダーフローが適切に処理されること.

    Given:
        - math.expのオーバーフローを引き起こす可能性のある極端な入力値がある
        - 非常に大きい正の値（1e10）がある
        - 非常に大きい負の値（-1e10）がある
    When:
        - 極端な値でsigmoidが計算される
    Then:
        - 極端な正の値に対して1.0が返されること（例外なし）
        - 極端な負の値に対して0.0が返されること（例外なし）
        - OverflowErrorまたはアンダーフロー例外が発生しないこと
    """
    # Arrange
    center = 500.0
    extreme_positive = 1e10
    extreme_negative = -1e10

    # Act
    result_positive = MetricNormalizer.sigmoid(extreme_positive, center)
    result_negative = MetricNormalizer.sigmoid(extreme_negative, center)

    # Assert
    # 境界値を返すことで適切に処理するはず
    assert result_positive == 1.0
    assert result_negative == 0.0


def test_normalize_all_returns_all_expected_metrics() -> None:
    """すべての8つの期待される正規化メトリックを含む辞書が返されること.

    Given:
        - すべての必須フィールドを含む生メトリック辞書がある
    When:
        - normalize_allが呼び出される
    Then:
        - すべての8つの期待されるメトリックが返されること：
          - blur_score
          - contrast
          - color_richness
          - edge_density
          - dramatic_score
          - visual_balance
          - action_intensity
          - ui_density
    """
    # Arrange
    raw = {
        "blur_score": 500.0,
        "contrast": 50.0,
        "color_richness": 40.0,
        "edge_density": 0.2,
        "dramatic_score": 50.0,
        "visual_balance": 80.0,
        "action_intensity": 30.0,
        "ui_density": 10.0,
    }

    # Act
    result = MetricNormalizer.normalize_all(raw)

    # Assert
    assert len(result) == 8
    expected_keys = {
        "blur_score",
        "contrast",
        "color_richness",
        "edge_density",
        "dramatic_score",
        "visual_balance",
        "action_intensity",
        "ui_density",
    }
    assert set(result.keys()) == expected_keys


def test_normalize_all_produces_values_in_valid_range() -> None:
    """正規化値が有効範囲内にあること.

    Given:
        - 有効な生メトリクスがある
    When:
        - normalize_allが呼び出される
    Then:
        - すべての正規化値が[0, 1]範囲内にあること
    """
    # Arrange
    raw = {
        "blur_score": 500.0,
        "contrast": 50.0,
        "color_richness": 40.0,
        "edge_density": 0.2,
        "dramatic_score": 50.0,
        "visual_balance": 80.0,
        "action_intensity": 30.0,
        "ui_density": 10.0,
    }

    # Act
    result = MetricNormalizer.normalize_all(raw)

    # Assert
    for value in result.values():
        assert 0.0 <= value <= 1.0
