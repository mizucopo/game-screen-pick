"""CLI実行時に使うTOML設定ローダー."""

import logging
import tomllib
from pathlib import Path
from typing import Any

from ..models.scene_mix import SceneMix

logger = logging.getLogger(__name__)


class ConfigLoader:
    """TOML設定をロードする.

    サポート対象はCLIで上書き可能なキーだけに限定し、
    設定ファイルから読み出した値を `SelectionConfig` 用の
    部分辞書へ変換する責務だけを持つ。
    """

    @staticmethod
    def load(path: str | None) -> dict[str, Any]:
        """TOML設定を辞書で返す.

        `[selection]`, `[scene_mix]`, `[thresholds]` のうち、
        実行時に必要な項目だけを抽出して返す。
        優先順位の解決自体は行わず、後段の `Main.build_selection_config`
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

        KNOWN_SECTIONS = {"selection", "scene_mix", "thresholds"}
        for section_name in raw_data:
            if section_name not in KNOWN_SECTIONS:
                logger.warning(f"未知のセクションを無視しました: [{section_name}]")

        result: dict[str, Any] = {}
        selection = raw_data.get("selection", {})
        KNOWN_SELECTION_KEYS = {"profile"}
        for key in selection:
            if key not in KNOWN_SELECTION_KEYS:
                logger.warning(f"未知のキーを無視しました: [selection] {key}")
        if "profile" in selection:
            result["profile"] = selection["profile"]

        scene_mix = raw_data.get("scene_mix", {})
        if scene_mix:
            result["scene_mix"] = SceneMix(
                play=float(scene_mix.get("play", 0.7)),
                event=float(scene_mix.get("event", 0.3)),
            )

        thresholds = raw_data.get("thresholds", {})
        if "similarity" in thresholds:
            result["similarity_threshold"] = float(thresholds["similarity"])

        return result
