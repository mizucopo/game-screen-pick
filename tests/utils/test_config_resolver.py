"""ConfigResolver の単体テスト."""

from pathlib import Path

import pytest

from src.utils.config_resolver import ConfigResolver


def test_resolve_selection_config_prefers_cli_over_config_file(
    tmp_path: Path,
) -> None:
    """CLI上書き値が設定ファイル値より優先されること.

    Arrange:
        - 設定ファイルに thresholds / ollama が設定されている
        - CLI上書き値に similarity / ollama が指定されている
    Act:
        - SelectionConfig が解決される
    Assert:
        - CLI値が設定ファイル値より優先されること
    """
    # Arrange
    config_path = tmp_path / "picker.toml"
    config_path.write_text(
        "[thresholds]\nsimilarity = 0.66\n"
        '[ollama]\nmodel = "config-model"\nhost = "http://config:11434"\n'
        "timeout = 90\nmax_workers = 3\n",
        encoding="utf-8",
    )

    # Act
    config = ConfigResolver.resolve_selection_config(
        config_path=str(config_path),
        similarity=0.8,
        batch_size=None,
        ollama_model="cli-model",
        ollama_host="http://cli:11434",
        ollama_timeout=30.0,
        ollama_max_workers=2,
        ollama_cache_enabled=False,
        scene_hint="RPG。戦闘と探索が混在している",
    )

    # Assert
    assert config.similarity_threshold == 0.8
    assert config.ollama is not None
    assert config.ollama.model == "cli-model"
    assert config.ollama.host == "http://cli:11434"
    assert config.ollama.timeout == 30.0
    assert config.ollama.max_workers == 2
    assert config.ollama.cache_enabled is False
    assert config.scene_hint == "RPG。戦闘と探索が混在している"


def test_resolve_selection_config_requires_ollama_model() -> None:
    """Ollamaモデル未指定の場合は失敗すること.

    Arrange:
        - 設定ファイルとCLI上書き値にモデルがない
    Act:
        - SelectionConfig が解決される
    Assert:
        - モデル必須エラーになること
    """
    # Arrange
    config_path = None

    # Act / Assert
    with pytest.raises(ValueError, match="ollama_model"):
        ConfigResolver.resolve_selection_config(
            config_path=config_path,
            similarity=None,
            batch_size=None,
            ollama_model=None,
            ollama_host=None,
            ollama_timeout=None,
            ollama_max_workers=None,
            ollama_cache_enabled=True,
            scene_hint=None,
        )


def test_resolve_selection_config_prefers_environment_host(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """OLLAMA_HOST が設定ファイルのhostより優先されること.

    Arrange:
        - 設定ファイルと環境変数にhostがある
        - CLI hostは指定されていない
    Act:
        - SelectionConfig が解決される
    Assert:
        - OLLAMA_HOST が使用されること
    """
    # Arrange
    config_path = tmp_path / "picker.toml"
    config_path.write_text(
        '[ollama]\nmodel = "gemma4"\nhost = "http://config:11434"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("OLLAMA_HOST", "http://env:11434")

    # Act
    config = ConfigResolver.resolve_selection_config(
        config_path=str(config_path),
        similarity=None,
        batch_size=None,
        ollama_model=None,
        ollama_host=None,
        ollama_timeout=None,
        ollama_max_workers=None,
        ollama_cache_enabled=True,
        scene_hint=None,
    )

    # Assert
    assert config.ollama is not None
    assert config.ollama.host == "http://env:11434"


def test_resolve_configs_returns_analyzer_and_selection_configs_from_cli_values(
    tmp_path: Path,
) -> None:
    """AnalyzerConfig と SelectionConfig がCLI値から解決されること.

    Arrange:
        - 設定ファイルに thresholds 設定が指定されている
        - CLI上書き値に analyzer 設定と selection 設定が指定されている
    Act:
        - 実行時設定がまとめて解決される
    Assert:
        - analyzer 設定にCLI値が反映されること
        - selection 設定に優先順位どおりの値が反映されること
    """
    # Arrange
    config_path = tmp_path / "picker.toml"
    config_path.write_text(
        '[thresholds]\nsimilarity = 0.66\n[ollama]\nmodel = "config-model"\n',
        encoding="utf-8",
    )

    # Act
    analyzer_config, selection_config = ConfigResolver.resolve_configs(
        config_path=str(config_path),
        similarity=None,
        batch_size=64,
        result_max_workers=2,
        max_dim=1080,
        max_memory_gb=4,
        ollama_model=None,
        ollama_host=None,
        ollama_timeout=None,
        ollama_max_workers=None,
        ollama_cache_enabled=True,
        scene_hint=None,
    )

    # Assert
    assert analyzer_config.result_max_workers == 2
    assert analyzer_config.max_dim == 1080
    assert analyzer_config.max_memory_gb == 4
    assert selection_config.similarity_threshold == 0.66
    assert selection_config.batch_size == 64
    assert selection_config.ollama is not None
    assert selection_config.ollama.model == "config-model"
