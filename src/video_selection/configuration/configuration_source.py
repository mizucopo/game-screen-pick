"""Effective Configurationの値source。"""

from enum import StrEnum


class ConfigurationSource(StrEnum):
    """各設定値を採用した公開source。"""

    CLI = "cli"
    TOML = "toml"
    ENVIRONMENT = "environment"
    BUILT_IN = "built-in"
