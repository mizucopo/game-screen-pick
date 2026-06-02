"""実行時設定の解決."""

from typing import Any

from ..models.analyzer_config import AnalyzerConfig
from ..models.scene_mix import SceneMix
from ..models.selection_config import SelectionConfig
from .config_loader import ConfigLoader


class ConfigResolver:
    """設定ファイル値とCLI上書き値から実行時設定を構築する."""

    @staticmethod
    def resolve_configs(
        *,
        config_path: str | None,
        profile: str | None,
        scene_mix: SceneMix | None,
        similarity: float | None,
        batch_size: int | None,
        result_max_workers: int | None,
        max_dim: int,
        max_memory_gb: int,
    ) -> tuple[AnalyzerConfig, SelectionConfig]:
        """解析設定と選択設定を構築する."""
        analyzer_config = AnalyzerConfig.from_cli_args(
            result_max_workers=result_max_workers,
            max_dim=max_dim,
            max_memory_gb=max_memory_gb,
        )
        selection_config = ConfigResolver.resolve_selection_config(
            config_path=config_path,
            profile=profile,
            scene_mix=scene_mix,
            similarity=similarity,
            batch_size=batch_size,
        )
        return analyzer_config, selection_config

    @staticmethod
    def resolve_selection_config(
        *,
        config_path: str | None,
        profile: str | None,
        scene_mix: SceneMix | None,
        similarity: float | None,
        batch_size: int | None,
    ) -> SelectionConfig:
        """設定ファイル値とCLI上書き値から選択設定を構築する."""
        config_values = ConfigLoader.load(config_path)
        cli_overrides: dict[str, Any] = {
            "profile": profile,
            "scene_mix": scene_mix,
            "similarity_threshold": similarity,
            "batch_size": batch_size,
        }
        merged = {
            **config_values,
            **{key: value for key, value in cli_overrides.items() if value is not None},
        }
        return SelectionConfig.from_cli_args(**merged)
