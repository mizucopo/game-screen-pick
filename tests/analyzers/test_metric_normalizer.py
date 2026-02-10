"""MetricNormalizerの単体テスト.

このテストモジュールは以下のベストプラクティスに従っています：
1.「How」（実装詳細）ではなく「What」（観察可能な挙動）をテスト
2. モックなし - 予測可能な入出力を持つ純粋な関数テスト
3. 明確なコメント付きのAAAパターン（Arrange, Act, Assert）を使用
4. 高速実行（約0.1秒） - 外部依存関係なし
"""

from src.analyzers.metric_normalizer import MetricNormalizer


# ============================================================================
# sigmoid関数のテスト（5件）
# ============================================================================


def test_sigmoid_returns_0_5_when_x_equals_center() -> None:
    """xがcenterと等しい場合、sigmoidは正確に0.5を返す.

    Given:
        - center値が500.0
        - xがcenter値と等しい
    When:
        - sigmoid(x, center)を計算
    Then:
        - 正確に0.5を返す（sigmoid曲線の中点）
    """
    # Arrange
    center = 500.0
    x = center

    # Act
    result = MetricNormalizer.sigmoid(x, center)

    # Assert
    assert result == 0.5


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
# normalize_allメソッドのテスト（7件）
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


def test_normalize_all_applies_sigmoid_to_blur_score() -> None:
    """blur_scoreメトリックにsigmoid正規化を適用する.

    Given:
        - 生のblur_scoreが500.0（center値）
    When:
        - normalize_allを呼び出し
    Then:
        - blur_scoreはcenter=500のsigmoidを使用して正規化される
        - centerポイントで正確に0.5になる
    """
    # Arrange
    raw = {
        "blur_score": 500.0,
        "contrast": 50.0,
        "color_richness": 40.0,
        "edge_density": 0.1,
        "dramatic_score": 50.0,
        "visual_balance": 80.0,
        "action_intensity": 30.0,
        "ui_density": 10.0,
    }

    # Act
    result = MetricNormalizer.normalize_all(raw)

    # Assert
    assert result["blur_score"] == 0.5


def test_normalize_all_applies_sigmoid_to_contrast() -> None:
    """contrastメトリックにsigmoid正規化を適用する.

    Given:
        - 生のcontrastが50.0（center値）
    When:
        - normalize_allを呼び出し
    Then:
        - contrastはcenter=50のsigmoidを使用して正規化される
        - centerポイントで正確に0.5になる
    """
    # Arrange
    raw = {
        "blur_score": 500.0,
        "contrast": 50.0,
        "color_richness": 40.0,
        "edge_density": 0.1,
        "dramatic_score": 50.0,
        "visual_balance": 80.0,
        "action_intensity": 30.0,
        "ui_density": 10.0,
    }

    # Act
    result = MetricNormalizer.normalize_all(raw)

    # Assert
    assert result["contrast"] == 0.5


def test_normalize_all_clips_edge_density_to_max_1() -> None:
    """min(1.0, raw * 5.0)を使用してedge_densityを最大1.0にクリップする.

    Given:
        - 生のedge_densityが0.3（5倍すると1.5になる値）
    When:
        - normalize_allを呼び出し
    Then:
        - edge_densityは最大1.0にクリップされる
        - 公式：min(1.0, raw * 5.0)
    """
    # Arrange
    raw = {
        "blur_score": 500.0,
        "contrast": 50.0,
        "color_richness": 40.0,
        "edge_density": 0.3,  # 1.5になり、1.0にクリップされる
        "dramatic_score": 50.0,
        "visual_balance": 80.0,
        "action_intensity": 30.0,
        "ui_density": 10.0,
    }

    # Act
    result = MetricNormalizer.normalize_all(raw)

    # Assert
    assert result["edge_density"] == 1.0


def test_normalize_all_divides_visual_balance_by_100() -> None:
    """[0, 1]範囲に正規化するためvisual_balanceを100で割る.

    Given:
        - 生のvisual_balanceが80.0
    When:
        - normalize_allを呼び出し
    Then:
        - visual_balanceは100で割られる
        - 結果は0.8になる
    """
    # Arrange
    raw = {
        "blur_score": 500.0,
        "contrast": 50.0,
        "color_richness": 40.0,
        "edge_density": 0.1,
        "dramatic_score": 50.0,
        "visual_balance": 80.0,
        "action_intensity": 30.0,
        "ui_density": 10.0,
    }

    # Act
    result = MetricNormalizer.normalize_all(raw)

    # Assert
    assert result["visual_balance"] == 0.8


def test_normalize_all_produces_values_between_0_and_1() -> None:
    """すべての正規化値が有効範囲[0, 1]内にある.

    Given:
        - 様々な現実的な値を持つ生メトリック
    When:
        - normalize_allを呼び出し
    Then:
        - すべての正規化値が0.0から1.0の間（両端含む）
        - 負の値や1.0より大きい値は存在しない
    """
    # Arrange
    raw = {
        "blur_score": 650.0,
        "contrast": 75.0,
        "color_richness": 55.0,
        "edge_density": 0.25,
        "dramatic_score": 80.0,
        "visual_balance": 90.0,
        "action_intensity": 45.0,
        "ui_density": 15.0,
    }

    # Act
    result = MetricNormalizer.normalize_all(raw)

    # Assert
    for value in result.values():
        assert 0.0 <= value <= 1.0


def test_normalize_all_with_different_raw_values_produces_different_results() -> None:
    """異なる生の入力値は異なる正規化出力を生成する.

    Given:
        - 異なる値を持つ2セットの生メトリック
    When:
        - 各セットでnormalize_allを呼び出し
    Then:
        - 異なる正規化結果を生成する
        - 結果は相対的な順序を維持する（より高い生値 -> より高い正規化値）
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
        "edge_density": 0.2,
        "dramatic_score": 70.0,
        "visual_balance": 95.0,
        "action_intensity": 40.0,
        "ui_density": 15.0,
    }

    # Act
    result_low = MetricNormalizer.normalize_all(raw_low)
    result_high = MetricNormalizer.normalize_all(raw_high)

    # Assert
    # より高い生の値はより高い正規化値を生成するはず
    assert result_high["blur_score"] > result_low["blur_score"]
    assert result_high["contrast"] > result_low["contrast"]
    assert result_high["color_richness"] > result_low["color_richness"]
    # Edge densityは線形スケーリングを使用
    assert result_high["edge_density"] > result_low["edge_density"]
    # Visual balanceは線形スケーリングを使用
    assert result_high["visual_balance"] > result_low["visual_balance"]
