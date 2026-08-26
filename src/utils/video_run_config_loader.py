"""動画選定CLI向けTOML設定の読み込み."""

import tomllib
from pathlib import Path
from typing import Any


class VideoRunConfigLoader:
    """TOMLの `[run]` tableを厳格に読み込む."""

    _STRING_KEYS = {
        "game_context_provider",
        "game_context_model",
        "primary_model",
        "secondary_model",
        "ollama_host",
    }
    _INTEGER_KEYS = {"ffmpeg_workers"}
    _NUMBER_KEYS = {"ollama_timeout", "sample_interval_seconds"}
    _BOOLEAN_KEYS = {"allow_cpu", "debug"}
    _KNOWN_KEYS = _STRING_KEYS | _INTEGER_KEYS | _NUMBER_KEYS | _BOOLEAN_KEYS

    @classmethod
    def load(cls, path: str) -> dict[str, object]:
        """設定pathから検証済みの部分設定を返す."""
        config_path = Path(path)
        try:
            with config_path.open("rb") as file:
                raw_config = tomllib.load(file)
        except (OSError, tomllib.TOMLDecodeError) as error:
            msg = f"設定ファイルを読み込めません: {error}"
            raise ValueError(msg) from error

        unknown_sections = set(raw_config) - {"run"}
        if unknown_sections:
            section = sorted(unknown_sections)[0]
            msg = f"未知の設定セクションです: [{section}]"
            raise ValueError(msg)

        raw_run = raw_config.get("run", {})
        if not isinstance(raw_run, dict):
            msg = "[run] はtableで指定してください"
            raise ValueError(msg)

        unknown_keys = set(raw_run) - cls._KNOWN_KEYS
        if unknown_keys:
            key = sorted(unknown_keys)[0]
            msg = f"未知の設定キーです: [run].{key}"
            raise ValueError(msg)

        return {key: cls._validate_value(key, value) for key, value in raw_run.items()}

    @classmethod
    def _validate_value(cls, key: str, value: Any) -> object:
        """TOML値がkeyの公開型と一致することを検証する."""
        if key in cls._STRING_KEYS:
            if isinstance(value, str):
                return value
            expected_type = "string"
        elif key in cls._INTEGER_KEYS:
            if isinstance(value, int) and not isinstance(value, bool):
                return value
            expected_type = "integer"
        elif key in cls._NUMBER_KEYS:
            if isinstance(value, int | float) and not isinstance(value, bool):
                return float(value)
            expected_type = "number"
        elif key in cls._BOOLEAN_KEYS:
            if isinstance(value, bool):
                return value
            expected_type = "boolean"
        else:  # pragma: no cover - load()が未知keyを先に拒否する
            raise AssertionError(f"未定義の設定キーです: {key}")

        msg = f"[run].{key} は{expected_type}で指定してください"
        raise ValueError(msg)
