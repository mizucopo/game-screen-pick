"""SelectionConfigの単体テスト."""

import pytest

from src.models.selection_config import SelectionConfig


def test_selection_config_has_sensible_defaults() -> None:
    """SelectionConfigが合理的なデフォルト値を持つこと."""
    # Arrange & Act
    config = SelectionConfig()

    # Assert
    assert config.batch_size == 32
    assert config.max_threshold == 0.98
    assert len(config.threshold_relaxation_steps) > 0


@pytest.mark.parametrize(
    "base_threshold,max_threshold,expected_steps",
    [
        # 通常のステップ計算
        (0.72, 0.98, [0.72, 0.75, 0.78, 0.82, 0.87]),
        # 上限制限（max_thresholdでキャップされる）
        (0.95, 0.98, [0.95, 0.98, 0.98, 0.98, 0.98]),
    ],
)
def test_threshold_steps_computed_correctly(
    base_threshold: float,
    max_threshold: float,
    expected_steps: list[float],
) -> None:
    """しきい値ステップが正しく計算されること."""
    # Arrange
    config = SelectionConfig(max_threshold=max_threshold)

    # Act
    steps = config.compute_threshold_steps(base_threshold)

    # Assert
    assert steps == expected_steps


@pytest.mark.parametrize(
    "field_name,invalid_value",
    [
        ("batch_size", 0),
        ("batch_size", -10),
        ("max_threshold", -0.1),
        ("max_threshold", 1.1),
        ("threshold_relaxation_steps", [0.1, -0.05, 0.2]),
    ],
)
def test_invalid_values_rejected(
    field_name: str,
    invalid_value: int | float | list[float] | tuple[float, ...],
) -> None:
    """無効な値が設定された場合に例外が発生すること."""
    # Arrange & Act & Assert
    with pytest.raises(ValueError):
        SelectionConfig(**{field_name: invalid_value})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "activity_mix_ratio,should_raise",
    [
        # 有効な合計値（浮動小数点の誤差を許容）
        ((0.3, 0.4, 0.3), False),
        ((0.333, 0.333, 0.334), False),
        ((1.0, 0.0, 0.0), False),
        # 無効な合計値（1.0から大きくずれる）
        ((1.0, 1.0, 1.0), True),
        ((0.5, 0.5, 0.5), True),
        ((0.0, 0.0, 0.0), True),
    ],
)
def test_activity_mix_ratio_sum_validation(
    activity_mix_ratio: tuple[float, float, float],
    should_raise: bool,
) -> None:
    """activity_mix_ratioの合計値が1.0であることを検証すること."""
    # Arrange & Act & Assert
    if should_raise:
        with pytest.raises(ValueError, match="合計は1.0"):
            SelectionConfig(activity_mix_ratio=activity_mix_ratio)
    else:
        config = SelectionConfig(activity_mix_ratio=activity_mix_ratio)
        assert config.activity_mix_ratio == activity_mix_ratio
