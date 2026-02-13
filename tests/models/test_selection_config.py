"""SelectionConfigの単体テスト."""

import pytest

from src.models.selection_config import SelectionConfig


@pytest.mark.parametrize(
    "batch_size,steps,max_threshold,expected_batch,expected_steps,expected_max",
    [
        # デフォルト値
        (None, None, None, 32, [0.03, 0.06, 0.10, 0.15], 0.98),
        # カスタム値
        (64, [0.05, 0.10, 0.20], 0.95, 64, [0.05, 0.10, 0.20], 0.95),
    ],
)
def test_selection_config_initialization(
    batch_size: int | None,
    steps: list[float] | None,
    max_threshold: float | None,
    expected_batch: int,
    expected_steps: list[float],
    expected_max: float,
) -> None:
    """SelectionConfigが正しく初期化されること.

    Given:
        - デフォルト値またはカスタム値がある
    When:
        - SelectionConfigを作成
    Then:
        - 指定した値が正しく設定されていること
    """
    # Arrange & Act
    if batch_size is None:
        config = SelectionConfig()
    else:
        config = SelectionConfig(
            batch_size=batch_size,
            threshold_relaxation_steps=steps,  # type: ignore[arg-type]
            max_threshold=max_threshold,  # type: ignore[arg-type]
        )

    # Assert
    assert config.batch_size == expected_batch
    assert config.threshold_relaxation_steps == expected_steps
    assert config.max_threshold == expected_max


@pytest.mark.parametrize(
    "base_threshold,max_threshold,expected_steps",
    [
        # 通常のステップ計算
        (0.72, 0.98, [0.72, 0.75, 0.78, 0.82, 0.87]),
        # 上限制限（max_thresholdでキャップされる）
        (0.95, 0.98, [0.95, 0.98, 0.98, 0.98, 0.98]),
        # 低いベース値でのステップ計算
        (0.50, 0.98, [0.50, 0.53, 0.56, 0.60, 0.65]),
    ],
)
def test_compute_threshold_steps_with_defaults(
    base_threshold: float,
    max_threshold: float,
    expected_steps: list[float],
) -> None:
    """デフォルトステップでのしきい値計算と上限制限が正しく動作すること.

    Given:
        - デフォルトステップを持つSelectionConfigがある
        - ベースしきい値と最大しきい値がある
    When:
        - しきい値ステップを計算
    Then:
        - 期待されるステップが計算されること
        - 上限制限が遵守されること
    """
    # Arrange
    config = SelectionConfig(max_threshold=max_threshold)

    # Act
    steps = config.compute_threshold_steps(base_threshold)

    # Assert
    assert steps == expected_steps


def test_compute_threshold_steps_with_custom_steps() -> None:
    """カスタムステップが正しく適用されること.

    Given:
        - カスタムステップを持つSelectionConfigがある
    When:
        - しきい値ステップを計算
    Then:
        - カスタムステップがベース値に加算されること
    """
    # Arrange
    custom_steps = [0.05, 0.15]
    base_threshold = 0.70
    config = SelectionConfig(
        threshold_relaxation_steps=custom_steps,
        max_threshold=0.95,
    )

    # Act
    steps = config.compute_threshold_steps(base_threshold)

    # Assert
    assert steps == [0.70, 0.75, 0.85]


@pytest.mark.parametrize(
    "field_name,invalid_value,error_match",
    [
        ("batch_size", 0, None),
        ("batch_size", -10, None),
        ("max_threshold", -0.1, None),
        ("max_threshold", 1.1, None),
        ("threshold_relaxation_steps", [0.1, -0.05, 0.2], None),
        ("activity_mix_ratio", (-0.1, 0.5, 0.6), "0以上1以下"),
        ("activity_mix_ratio", (0.3, 0.4, 1.1), "0以上1以下"),
    ],
)
def test_selection_config_rejects_invalid_values(
    field_name: str,
    invalid_value: int | float | list[float] | tuple[float, ...],
    error_match: str | None,
) -> None:
    """無効な値が設定された場合に例外が発生すること.

    Given:
        - 無効な値
    When:
        - SelectionConfigを作成
    Then:
        - ValueErrorがスローされること
    """
    # Arrange & Act & Assert
    if error_match:
        with pytest.raises(ValueError, match=error_match):
            SelectionConfig(**{field_name: invalid_value})  # type: ignore[arg-type]
    else:
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
    """activity_mix_ratioの合計値が1.0であることを検証すること.

    Given:
        - 様々なactivity_mix_ratioの値
    When:
        - SelectionConfigを作成
    Then:
        - 合計が1.0（許容範囲内）の場合は成功
        - 合計が1.0から大きくずれる場合はValueError
    """
    # Arrange & Act & Assert
    if should_raise:
        with pytest.raises(ValueError, match="合計は1.0"):
            SelectionConfig(activity_mix_ratio=activity_mix_ratio)
    else:
        config = SelectionConfig(activity_mix_ratio=activity_mix_ratio)
        assert config.activity_mix_ratio == activity_mix_ratio
