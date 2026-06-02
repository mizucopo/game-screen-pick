"""ConfigResolver の単体テスト."""

from pathlib import Path

from src.models.scene_mix import SceneMix
from src.utils.config_resolver import ConfigResolver


def test_resolve_selection_config_prefers_cli_over_config_file(
    tmp_path: Path,
) -> None:
    """CLI上書き値が設定ファイル値より優先されること.

    Arrange:
        - 設定ファイルに profile / scene_mix / similarity が設定されている
        - CLI上書き値に profile / similarity が指定されている
    Act:
        - SelectionConfig が解決される
    Assert:
        - profile と similarity はCLI値が使用されること
        - scene_mix は設定ファイル値が使用されること
    """
    # Arrange
    config_path = tmp_path / "picker.toml"
    config_path.write_text(
        '[selection]\nprofile = "static"\n'
        "[scene_mix]\nplay = 0.6\nevent = 0.4\n"
        "[thresholds]\nsimilarity = 0.66\n",
        encoding="utf-8",
    )

    # Act
    config = ConfigResolver.resolve_selection_config(
        config_path=str(config_path),
        profile="active",
        scene_mix=None,
        similarity=0.8,
        batch_size=None,
    )

    # Assert
    assert config.profile == "active"
    assert config.similarity_threshold == 0.8
    assert config.scene_mix.play == 0.6
    assert config.scene_mix.event == 0.4


def test_resolve_selection_config_uses_defaults_when_values_are_absent() -> None:
    """未指定値に組み込みデフォルトが使用されること.

    Arrange:
        - 設定ファイルとCLI上書き値が指定されていない
    Act:
        - SelectionConfig が解決される
    Assert:
        - SelectionConfig のデフォルト値が返されること
    """
    # Arrange
    config_path = None

    # Act
    config = ConfigResolver.resolve_selection_config(
        config_path=config_path,
        profile=None,
        scene_mix=None,
        similarity=None,
        batch_size=None,
    )

    # Assert
    assert config.profile == "auto"
    assert config.similarity_threshold == 0.72
    assert config.scene_mix.play == 0.7
    assert config.scene_mix.event == 0.3
    assert config.batch_size == 32


def test_resolve_configs_returns_analyzer_and_selection_configs_from_cli_values(
    tmp_path: Path,
) -> None:
    """AnalyzerConfig と SelectionConfig がCLI値から解決されること.

    Arrange:
        - 設定ファイルに selection 設定が指定されている
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
        '[selection]\nprofile = "static"\n'
        "[scene_mix]\nplay = 0.6\nevent = 0.4\n"
        "[thresholds]\nsimilarity = 0.66\n",
        encoding="utf-8",
    )

    # Act
    analyzer_config, selection_config = ConfigResolver.resolve_configs(
        config_path=str(config_path),
        profile=None,
        scene_mix=SceneMix(play=0.8, event=0.2),
        similarity=None,
        batch_size=64,
        result_max_workers=2,
        max_dim=1080,
        max_memory_gb=4,
    )

    # Assert
    assert analyzer_config.result_max_workers == 2
    assert analyzer_config.max_dim == 1080
    assert analyzer_config.max_memory_gb == 4
    assert selection_config.profile == "static"
    assert selection_config.similarity_threshold == 0.66
    assert selection_config.scene_mix.play == 0.8
    assert selection_config.scene_mix.event == 0.2
    assert selection_config.batch_size == 64
