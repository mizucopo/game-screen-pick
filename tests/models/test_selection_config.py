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
    "field_name,invalid_value",
    [
        ("batch_size", 0),
        ("batch_size", -10),
        ("max_threshold", -0.1),
        ("max_threshold", 1.1),
        ("threshold_relaxation_steps", [0.1, -0.05, 0.2]),
    ],
)
def test_selection_config_rejects_invalid_values(
    field_name: str,
    invalid_value: int | float | list[float],
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
    with pytest.raises(ValueError):
        SelectionConfig(**{field_name: invalid_value})  # type: ignore[arg-type]
