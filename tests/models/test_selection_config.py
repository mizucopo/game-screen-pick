"""SelectionConfigの単体テスト.

scene mix 導入後の設定モデルについて、
デフォルト値、しきい値緩和、入力バリデーションの公開挙動を確認する。
"""

import pytest

from src.models.scene_mix import SceneMix
from src.models.selection_config import SelectionConfig


def test_selection_config_has_sensible_defaults() -> None:
    """SelectionConfigが合理的なデフォルト値を持つこと.

    Given:
        - 引数なしで `SelectionConfig` を構築する
    When:
        - デフォルト設定を参照する
    Then:
        - profile、similarity、scene mix を含む既定値が入っていること
    """
    # Arrange & Act
    config = SelectionConfig()

    # Assert
    assert config.batch_size == 32
    assert config.profile == "auto"
    assert config.similarity_threshold == 0.72
    assert config.scene_mix == SceneMix(gameplay=0.5, event=0.4, other=0.1)
    assert config.max_threshold == 0.98


@pytest.mark.parametrize(
    "base_threshold,max_threshold,expected_steps",
    [
        (0.72, 0.98, [0.72, 0.75, 0.78, 0.82, 0.87]),
        (0.95, 0.98, [0.95, 0.98, 0.98, 0.98, 0.98]),
    ],
)
def test_threshold_steps_computed_correctly(
    base_threshold: float,
    max_threshold: float,
    expected_steps: list[float],
) -> None:
    """しきい値ステップが正しく計算されること.

    Given:
        - ベースしきい値と上限値がある
    When:
        - 緩和ステップを計算する
    Then:
        - 上限を超えない段階的なしきい値列が返ること
    """
    # Arrange
    config = SelectionConfig(max_threshold=max_threshold)

    # Act / Assert
    assert config.compute_threshold_steps(base_threshold) == expected_steps


@pytest.mark.parametrize(
    "kwargs,expected_error,match_pattern",
    [
        ({"batch_size": 0}, ValueError, "正の整数"),
        ({"profile": "unknown"}, ValueError, "profile"),
        ({"similarity_threshold": 1.1}, ValueError, "0以上1以下"),
        ({"max_threshold": -0.1}, ValueError, "0以上1以下"),
        (
            {"scene_mix": SceneMix(gameplay=1.0, event=0.0, other=0.0)},
            None,
            None,
        ),
    ],
)
def test_config_validation(
    kwargs: dict[str, object],
    expected_error: type[Exception] | None,
    match_pattern: str | None,
) -> None:
    """設定値のバリデーションが正しく動作すること.

    Given:
        - 有効または無効な設定引数がある
    When:
        - `SelectionConfig` を構築する
    Then:
        - 無効値では例外、有効値ではそのまま保持されること
    """
    # Arrange / Act / Assert
    if expected_error:
        with pytest.raises(expected_error, match=match_pattern):
            SelectionConfig(**kwargs)  # type: ignore[arg-type]
    else:
        config = SelectionConfig(**kwargs)  # type: ignore[arg-type]
        for key, value in kwargs.items():
            assert getattr(config, key) == value


def test_scene_mix_validation_rejects_invalid_total() -> None:
    """scene_mixの合計が1.0でない場合は失敗すること.

    Given:
        - 合計が1.0にならない scene mix 比率がある
    When:
        - `SceneMix` を構築する
    Then:
        - 合計値バリデーションで失敗すること
    """
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="scene_mixの合計"):
        SceneMix(gameplay=0.6, event=0.3, other=0.3)
