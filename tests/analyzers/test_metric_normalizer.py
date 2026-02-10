"""MetricNormalizerの単体テスト.

このテストモジュールは以下のベストプラクティスに従っています：
1.「How」（実装詳細）ではなく「What」（観察可能な挙動）をテスト
2. モックなし - 予測可能な入出力を持つ純粋な関数テスト
3. 明確なコメント付きのAAAパターン（Arrange, Act, Assert）を使用
4. 高速実行（約0.1秒） - 外部依存関係なし
"""

from src.analyzers.metric_normalizer import MetricNormalizer


# ============================================================================
# sigmoid関数のテスト
# ============================================================================


def test_sigmoid_returns_high_values_when_x_above_center() -> None:
    """xがcenterより大きい場合、sigmoidは0.5より大きい値を返す.

    Given:
        - center値が500.0
        - xがcenterより中程度上（600.0）
    When:
        - デフォルトの急峻さでsigmoid(x, center)を計算
    Then:
        - 0.5より大きい値を返す
        - 非常に大きいxでは1.0に近づく（極端な値では1.0になる可能性あり）
    """
    # Arrange
    center = 500.0
    x = 600.0  # センター値よりやや高いが、極端ではない値

    # Act
    result = MetricNormalizer.sigmoid(x, center)

    # Assert
    assert result > 0.5
    assert result <= 1.0  # 極端な値では正確に1.0になる可能性がある


def test_sigmoid_returns_low_values_when_x_below_center() -> None:
    """xがcenterより小さい場合、sigmoidは0.5より小さい値を返す.

    Given:
        - center値が500.0
        - xがcenterより大幅に下（100.0）
    When:
        - sigmoid(x, center)を計算
    Then:
        - 0.5より小さい値を返す
        - 非常に小さいxでは0.0に近づく
    """
    # Arrange
    center = 500.0
    x = 100.0

    # Act
    result = MetricNormalizer.sigmoid(x, center)

    # Assert
    assert result < 0.5
    assert result > 0.0


def test_sigmoid_with_default_steepness_produces_expected_curve() -> None:
    """デフォルトの急峻さで期待されるsigmoid曲線を生成する.

    Given:
        - center値が50.0（コントラストメトリックの典型的な値）
        - デフォルトの急峻さが0.1
        - 3つのテストポイント：centerの下、center、centerの上
    When:
        - 各ポイントのsigmoidを計算
    Then:
        - 期待される値を持つ滑らかな曲線を生成
        - より小さい値はより小さい出力を生成
        - より大きい値はより大きい出力を生成
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
    """例外を発生させずにオーバーフロー/アンダーフローを適切に処理する.

    Given:
        - math.expのオーバーフローを引き起こす可能性のある極端な入力値
        - 非常に大きい正の値（1e10）
        - 非常に大きい負の値（-1e10）
    When:
        - 極端な値でsigmoidを計算
    Then:
        - 極端な正の値に対して1.0を返す（例外なし）
        - 極端な負の値に対して0.0を返す（例外なし）
        - OverflowErrorまたはアンダーフロー例外は発生しない
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


# ============================================================================
# normalize_allメソッドのテスト
# ============================================================================


def test_normalize_all_returns_all_expected_metrics() -> None:
    """すべての8つの期待される正規化メトリックを含む辞書を返す.

    Given:
        - すべての必須フィールドを含む生メトリック辞書
    When:
        - normalize_allを呼び出し
    Then:
        - すべての8つの期待されるメトリックを返す：
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


def test_normalize_all_produces_valid_and_unique_results() -> None:
    """正規化値が有効範囲内にあり、異なる入力値は異なる結果を生成する.

    Given:
        - 2セットの異なる生メトリック
        - 低い値セットと高い値セット
    When:
        - 各セットでnormalize_allを呼び出し
    Then:
        - すべての正規化値が[0, 1]範囲内にある
        - より高い生値はより高い正規化値を生成する
        - 結果は一意である
    """
    # Arrange
    raw_low = {
        "blur_score": 300.0,
        "contrast": 30.0,
        "color_richness": 25.0,
        "edge_density": 0.1,
        "dramatic_score": 30.0,
        "visual_balance": 60.0,
        "action_intensity": 20.0,
        "ui_density": 5.0,
    }

    raw_high = {
        "blur_score": 700.0,
        "contrast": 70.0,
        "color_richness": 55.0,
        "edge_density": 0.3,
        "dramatic_score": 80.0,
        "visual_balance": 90.0,
        "action_intensity": 45.0,
        "ui_density": 15.0,
    }

    # Act
    result_low = MetricNormalizer.normalize_all(raw_low)
    result_high = MetricNormalizer.normalize_all(raw_high)

    # Assert
    # すべての値が[0, 1]範囲内
    for value in result_low.values():
        assert 0.0 <= value <= 1.0
    for value in result_high.values():
        assert 0.0 <= value <= 1.0
    # より高い生の値はより高い正規化値を生成する
    assert result_high["blur_score"] > result_low["blur_score"]
    assert result_high["contrast"] > result_low["contrast"]
    assert result_high["color_richness"] > result_low["color_richness"]
    assert result_high["edge_density"] >= result_low["edge_density"]  # クリップ可能
    assert result_high["visual_balance"] > result_low["visual_balance"]
