"""MetricNormalizerの単体テスト."""

import pytest

from src.analyzers.metric_normalizer import MetricNormalizer
from src.models.normalized_metrics import NormalizedMetrics
from src.models.raw_metrics import RawMetrics


@pytest.mark.parametrize(
    "x,center,steepness,expected",
    [
        # center値の場合は0.5になる（シグモイドの特性）
        (500, 500, 0.1, 0.5),
        # centerより大きい値は0.5より大きくなる
        (510, 500, 0.1, pytest.approx(0.73, abs=0.01)),
        # centerより小さい値は0.5より小さくなる
        (490, 500, 0.1, pytest.approx(0.27, abs=0.01)),
        # 境界値：非常に大きな値（オーバーフロー対策）
        (1000000, 500, 0.1, 1.0),
        # 境界値：非常に小さい値（オーバーフロー対策）
        (-1000000, 500, 0.1, 0.0),
    ],
)
def test_sigmoid_returns_correct_values(
    x: float,
    center: float,
    steepness: float,
    expected: float,
) -> None:
    """シグモイド関数が正しい値を返すこと.

    Arrange:
        - 様々な入力値とcenter値
    Act:
        - sigmoidが呼び出される
    Assert:
        - 期待値に近い値が返されること
        - center値では0.5になること
        - オーバーフロー時は境界値0.0または1.0が返されること
    """
    # Arrange & Act
    result = MetricNormalizer.sigmoid(x, center, steepness)

    # Assert
    if isinstance(expected, float) and 0.0 <= expected <= 1.0:
        assert result == pytest.approx(expected, abs=0.01)
    else:
        assert result == expected


def test_normalize_all_returns_expected_values() -> None:
    """正規化が期待通りにスコアを返すこと.

    Arrange:
        - center値や境界値に設定された生メトリクスがある
    Act:
        - normalize_allが呼び出される
    Assert:
        - シグモイド正規化フィールドはcenter値で約0.5になること
        - 線形正規化フィールドは期待値になること
    """
    # Arrange
    raw = RawMetrics(
        blur_score=500.0,  # center=500 → 約0.5
        brightness=100.0,
        contrast=50.0,  # center=50 → 約0.5
        color_richness=40.0,  # center=40 → 約0.5
        edge_density=0.2,  # 0.2 * 5.0 = 1.0
        dramatic_score=50.0,  # 50 / 100 = 0.5
        visual_balance=80.0,  # 80 / 100 = 0.8
        action_intensity=30.0,  # center=30 → 約0.5
        ui_density=10.0,  # center=10 → 約0.5
        luminance_entropy=1.0,
        luminance_range=50.0,
        near_black_ratio=0.0,
        near_white_ratio=0.0,
        dominant_tone_ratio=0.5,
    )

    # Act
    result = MetricNormalizer.normalize_all(raw)

    # Assert
    assert isinstance(result, NormalizedMetrics)
    assert 0.0 <= result.blur_score <= 1.0
    assert result.blur_score == pytest.approx(0.5, abs=0.01)
    assert result.contrast == pytest.approx(0.5, abs=0.01)
    assert result.color_richness == pytest.approx(0.5, abs=0.01)
    assert result.edge_density == 1.0  # min(1.0, 0.2 * 5.0)
    assert result.dramatic_score == 0.5  # 50 / 100
    assert result.visual_balance == 0.8  # 80 / 100
    assert result.action_intensity == pytest.approx(0.5, abs=0.01)
    assert result.ui_density == pytest.approx(0.5, abs=0.01)
