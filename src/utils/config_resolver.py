"""実行時設定の解決."""

import os

from ..models.analyzer_config import AnalyzerConfig
from ..models.ollama_config import OllamaConfig
from ..models.selection_config import SelectionConfig
from .config_loader import ConfigLoader


class ConfigResolver:
    """設定ファイル値とCLI上書き値から実行時設定を構築する."""

    @staticmethod
    def resolve_configs(
        *,
        config_path: str | None,
        profile: str | None,
        similarity: float | None,
        batch_size: int | None,
        result_max_workers: int | None,
        max_dim: int,
        max_memory_gb: int,
        ollama_model: str | None,
        ollama_host: str | None,
        ollama_timeout: float | None,
        ollama_max_workers: int | None,
        ollama_cache_enabled: bool,
        scene_hint: str | None,
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
            similarity=similarity,
            batch_size=batch_size,
            ollama_model=ollama_model,
            ollama_host=ollama_host,
            ollama_timeout=ollama_timeout,
            ollama_max_workers=ollama_max_workers,
            ollama_cache_enabled=ollama_cache_enabled,
            scene_hint=scene_hint,
        )
        return analyzer_config, selection_config

    @staticmethod
    def resolve_selection_config(
        *,
        config_path: str | None,
        profile: str | None,
        similarity: float | None,
        batch_size: int | None,
        ollama_model: str | None,
        ollama_host: str | None,
        ollama_timeout: float | None,
        ollama_max_workers: int | None,
        ollama_cache_enabled: bool,
        scene_hint: str | None,
    ) -> SelectionConfig:
        """設定ファイル値とCLI上書き値から選択設定を構築する."""
        config_values = ConfigLoader.load(config_path)
        resolved_ollama = ConfigResolver._resolve_ollama_config(
            config_values=config_values,
            ollama_model=ollama_model,
            ollama_host=ollama_host,
            ollama_timeout=ollama_timeout,
            ollama_max_workers=ollama_max_workers,
            ollama_cache_enabled=ollama_cache_enabled,
        )
        cli_overrides = {
            "profile": profile,
            "similarity_threshold": similarity,
            "batch_size": batch_size,
            "scene_hint": scene_hint,
        }
        merged = dict(config_values)
        for key in (
            "ollama_model",
            "ollama_host",
            "ollama_timeout",
            "ollama_max_workers",
        ):
            merged.pop(key, None)
        for key, value in cli_overrides.items():
            if value is not None:
                merged[key] = value
        merged["ollama"] = resolved_ollama
        return SelectionConfig.from_cli_args(**merged)

    @staticmethod
    def _resolve_ollama_config(
        *,
        config_values: dict[str, object],
        ollama_model: str | None,
        ollama_host: str | None,
        ollama_timeout: float | None,
        ollama_max_workers: int | None,
        ollama_cache_enabled: bool,
    ) -> OllamaConfig:
        """Ollama設定を優先順位に従って解決する."""
        model = ollama_model or ConfigResolver._string_config_value(
            config_values, "ollama_model"
        )
        if model is None:
            msg = "ollama_modelは必須です"
            raise ValueError(msg)

        host = (
            ollama_host
            or os.environ.get("OLLAMA_HOST")
            or ConfigResolver._string_config_value(config_values, "ollama_host")
            or "http://localhost:11434"
        )
        timeout = (
            ollama_timeout
            if ollama_timeout is not None
            else ConfigResolver._float_config_value(
                config_values,
                "ollama_timeout",
                60.0,
            )
        )
        max_workers = (
            ollama_max_workers
            if ollama_max_workers is not None
            else ConfigResolver._int_config_value(
                config_values,
                "ollama_max_workers",
                1,
            )
        )
        return OllamaConfig(
            model=model,
            host=host,
            timeout=timeout,
            max_workers=max_workers,
            cache_enabled=ollama_cache_enabled,
        )

    @staticmethod
    def _string_config_value(
        config_values: dict[str, object],
        key: str,
    ) -> str | None:
        """設定辞書から文字列値を取得する."""
        value = config_values.get(key)
        if value is None:
            return None
        return str(value)

    @staticmethod
    def _float_config_value(
        config_values: dict[str, object],
        key: str,
        default: float,
    ) -> float:
        """設定辞書から浮動小数点値を取得する."""
        value = config_values.get(key)
        if value is None:
            return default
        if isinstance(value, int | float | str):
            return float(value)
        return default

    @staticmethod
    def _int_config_value(
        config_values: dict[str, object],
        key: str,
        default: int,
    ) -> int:
        """設定辞書から整数値を取得する."""
        value = config_values.get(key)
        if value is None:
            return default
        if isinstance(value, int | float | str):
            return int(value)
        return default
