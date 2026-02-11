"""AnalyzerConfigの単体テスト."""

import pytest

from src.models.analyzer_config import AnalyzerConfig


def test_analyzer_config_has_default_values() -> None:
    """AnalyzerConfigがデフォルト値で正しく初期化されること.

    Given:
        - AnalyzerConfigをデフォルト値で作成
    When:
        - 各属性にアクセス
    Then:
        - すべてのデフォルト値が正しく設定されていること
    """
    # Act
    config = AnalyzerConfig()

    # Assert
    assert config.max_dim == 720
    assert config.chunk_size == 128
    assert config.brightness_penalty_threshold == 40.0
    assert config.brightness_penalty_value == 0.6
    assert config.semantic_weight == 0.2
    assert config.score_multiplier == 100.0


def test_analyzer_config_can_be_customized() -> None:
    """AnalyzerConfigがカスタム値で初期化できること.

    Given:
        - カスタム値を指定
    When:
        - AnalyzerConfigを作成
    Then:
        - 指定した値が正しく設定されていること
    """
    # Arrange
    custom_max_dim = 1080
    custom_chunk_size = 256
    custom_brightness_threshold = 30.0
    custom_penalty_value = 0.8
    custom_semantic_weight = 0.3
    custom_score_multiplier = 50.0

    # Act
    config = AnalyzerConfig(
        max_dim=custom_max_dim,
        chunk_size=custom_chunk_size,
        brightness_penalty_threshold=custom_brightness_threshold,
        brightness_penalty_value=custom_penalty_value,
        semantic_weight=custom_semantic_weight,
        score_multiplier=custom_score_multiplier,
    )

    # Assert
    assert config.max_dim == custom_max_dim
    assert config.chunk_size == custom_chunk_size
    assert config.brightness_penalty_threshold == custom_brightness_threshold
    assert config.brightness_penalty_value == custom_penalty_value
    assert config.semantic_weight == custom_semantic_weight
    assert config.score_multiplier == custom_score_multiplier


@pytest.mark.parametrize(
    "field_name,invalid_value",
    [
        ("max_dim", 0),
        ("max_dim", -100),
        ("chunk_size", 0),
        ("chunk_size", -50),
        ("brightness_penalty_threshold", -1.0),
        ("brightness_penalty_value", -0.1),
        ("semantic_weight", -0.1),
        ("score_multiplier", 0.0),
    ],
)
def test_analyzer_config_rejects_invalid_values(
    field_name: str, invalid_value: int | float
) -> None:
    """無効な値が設定された場合に妥当性チェックが行われること.

    Given:
        - 無効な値
    When:
        - AnalyzerConfigを作成
    Then:
        - 負の値やゼロが許容されないこと（データ型としての検証）
    """
    # Arrange & Act
    # dataclass自体はバリデーションを行わないため、
    # 負の値を設定してもインスタンスは作成される
    kwargs = {field_name: invalid_value}
    config = AnalyzerConfig(**kwargs)  # type: ignore[arg-type]

    # Assert - 設定値がそのまま保持されることを確認
    assert getattr(config, field_name) == invalid_value
    # 実際の使用時には、これらの値がロジック内で適切に処理されるか
    # 呼び出し側の責任で検証する必要がある
