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
    "kwargs,expected_error,match_pattern",
    [
        # 数値範囲のバリデーション
        ({"batch_size": 0}, ValueError, "正の整数"),
        ({"batch_size": -10}, ValueError, "正の整数"),
        ({"max_threshold": -0.1}, ValueError, "0以上1以下"),
        ({"max_threshold": 1.1}, ValueError, "0以上1以下"),
        ({"threshold_relaxation_steps": [0.1, -0.05, 0.2]}, ValueError, "非負"),
        # activity_mix_ratioの合計値バリデーション
        ({"activity_mix_ratio": (1.0, 1.0, 1.0)}, ValueError, "合計は1.0"),
        # 有効な値（例外が発生しないことを確認）
        ({"activity_mix_ratio": (0.3, 0.4, 0.3)}, None, None),
    ],
)
def test_config_validation(
    kwargs: dict[str, object],
    expected_error: type[Exception] | None,
    match_pattern: str | None,
) -> None:
    """設定値のバリデーションが正しく動作すること."""
    # Arrange & Act & Assert
    if expected_error:
        if match_pattern:
            ctx = pytest.raises(expected_error, match=match_pattern)
        else:
            ctx = pytest.raises(expected_error)
        with ctx:
            SelectionConfig(**kwargs)  # type: ignore[arg-type]
    else:
        config = SelectionConfig(**kwargs)  # type: ignore[arg-type]
        for key, value in kwargs.items():
            assert getattr(config, key) == value
