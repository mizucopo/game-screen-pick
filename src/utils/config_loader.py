"""CLI実行時に使うTOML設定ローダー."""

import logging
import tomllib
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ConfigLoader:
    """TOML設定をロードする.

    サポート対象はCLIで上書き可能なキーだけに限定し、
    設定ファイルから読み出した値を `SelectionConfig` 用の
    部分辞書へ変換する責務だけを持つ。
    """

    KNOWN_SECTIONS = {"selection", "thresholds", "ollama"}
    KNOWN_SELECTION_KEYS: set[str] = set()
    KNOWN_OLLAMA_KEYS = {"model", "host", "timeout", "max_workers"}

    @staticmethod
    def load(path: str | None) -> dict[str, Any]:
        """TOML設定を辞書で返す.

        `[selection]`, `[thresholds]`, `[ollama]` のうち、
        実行時に必要な項目だけを抽出して返す。
        優先順位の解決自体は行わず、後段の `ConfigResolver`
        で CLI override と合成する前提の補助メソッドである。

        Args:
            path: 読み込むTOMLファイルのパス。 `None` の場合は空辞書を返す。

        Returns:
            `SelectionConfig.from_cli_args` へ渡せる部分設定辞書。
        """
        if path is None:
            return {}
        config_path = Path(path)
        with config_path.open("rb") as file:
            raw_data = tomllib.load(file)

        ConfigLoader._warn_unknown_sections(raw_data)

        result: dict[str, Any] = {}
        ConfigLoader._warn_unknown_keys(
            section_name="selection",
            values=raw_data.get("selection", {}),
            known_keys=ConfigLoader.KNOWN_SELECTION_KEYS,
        )

        thresholds = raw_data.get("thresholds", {})
        if "similarity" in thresholds:
            result["similarity_threshold"] = float(thresholds["similarity"])

        ollama = raw_data.get("ollama", {})
        ConfigLoader._warn_unknown_keys(
            section_name="ollama",
            values=ollama,
            known_keys=ConfigLoader.KNOWN_OLLAMA_KEYS,
        )
        if "model" in ollama:
            result["ollama_model"] = str(ollama["model"])
        if "host" in ollama:
            result["ollama_host"] = str(ollama["host"])
        if "timeout" in ollama:
            result["ollama_timeout"] = float(ollama["timeout"])
        if "max_workers" in ollama:
            result["ollama_max_workers"] = int(ollama["max_workers"])

        return result

    @staticmethod
    def _warn_unknown_sections(raw_data: dict[str, Any]) -> None:
        """未対応セクションを警告する."""
        for section_name in raw_data:
            if section_name not in ConfigLoader.KNOWN_SECTIONS:
                logger.warning(f"未知のセクションを無視しました: [{section_name}]")

    @staticmethod
    def _warn_unknown_keys(
        *,
        section_name: str,
        values: dict[str, Any],
        known_keys: set[str],
    ) -> None:
        """未対応キーを警告する."""
        for key in values:
            if key not in known_keys:
                logger.warning(f"未知のキーを無視しました: [{section_name}] {key}")
