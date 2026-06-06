"""SelectionConfigの単体テスト."""

import pytest

from src.models.ollama_config import OllamaConfig
from src.models.selection_config import SelectionConfig


def test_selection_config_has_sensible_defaults() -> None:
    """SelectionConfigが合理的なデフォルト値を持つこと.

    Arrange:
        - 引数なしで `SelectionConfig` を構築する
    Act:
        - デフォルト設定を参照する
    Assert:
        - similarity、Ollama関連を含む既定値が入っていること
    """
    # Arrange
    # (引数なしでデフォルト設定を使用)

    # Act
    config = SelectionConfig()

    # Assert
    assert config.batch_size == 32
    assert config.similarity_threshold == 0.72
    assert config.ollama is None
    assert config.scene_hint is None
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

    Arrange:
        - ベースしきい値と上限値がある
    Act:
        - 緩和ステップを計算する
    Assert:
        - 上限を超えない段階的なしきい値列が返ること
    """
    # Arrange
    config = SelectionConfig(max_threshold=max_threshold)

    # Act
    steps = config.compute_threshold_steps(base_threshold)

    # Assert
    assert steps == expected_steps


@pytest.mark.parametrize(
    "kwargs,expected_error,match_pattern",
    [
        ({"batch_size": 0}, ValueError, "正の整数"),
        ({"similarity_threshold": 1.1}, ValueError, "0以上1以下"),
        ({"max_threshold": -0.1}, ValueError, "0以上1以下"),
        (
            {"ollama": OllamaConfig(model="gemma4")},
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

    Arrange:
        - 有効または無効な設定引数がある
    Act:
        - `SelectionConfig` を構築する
    Assert:
        - 無効値では例外、有効値ではそのまま保持されること
    """
    # Arrange
    # (パラメータ化されたkwargsを使用)

    # Act / Assert
    if expected_error:
        with pytest.raises(expected_error, match=match_pattern):
            SelectionConfig(**kwargs)  # type: ignore[arg-type]
    else:
        config = SelectionConfig(**kwargs)  # type: ignore[arg-type]
        for key, value in kwargs.items():
            assert getattr(config, key) == value


def test_ollama_config_validation_rejects_empty_model() -> None:
    """Ollamaモデル名が空の場合は失敗すること.

    Arrange:
        - 空のモデル名がある
    Act:
        - `OllamaConfig` を構築する
    Assert:
        - モデル名必須のバリデーションで失敗すること
    """
    # Act / Assert
    with pytest.raises(ValueError, match="ollama_model"):
        OllamaConfig(model="")


@pytest.mark.parametrize(
    "input_host,expected_host",
    [
        ("192.168.1.31", "http://192.168.1.31:11434"),
        ("192.168.1.31:11435", "http://192.168.1.31:11435"),
        ("http://ollama", "http://ollama"),
        ("http://ollama:11435", "http://ollama:11435"),
    ],
)
def test_ollama_config_normalizes_scheme_less_host(
    input_host: str,
    expected_host: str,
) -> None:
    """schemeなしのOllama hostにHTTP schemeと既定portが補完されること.

    Arrange:
        - Ollama hostの入力値がある
    Act:
        - `OllamaConfig` が構築される
    Assert:
        - schemeなしhostにはHTTP schemeと必要な場合だけ既定portが補完されること
    """
    # Act
    config = OllamaConfig(model="gemma4", host=input_host)

    # Assert
    assert config.host == expected_host
