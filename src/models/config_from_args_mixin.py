"""設定クラス用の共通メソッド."""

from typing import Any, TypeVar

T = TypeVar("T")


class ConfigFromArgsMixin:
    """CLI引数から設定を作成するためのミックスイン."""

    @classmethod
    def from_cli_args(cls: type[T], **kwargs: Any) -> T:
        """CLI引数から設定を作成する.

        Args:
            **kwargs: CLI引数（Noneでない引数のみデフォルト値を上書き）

        Returns:
            設定クラスのインスタンス
        """
        filtered = {k: v for k, v in kwargs.items() if v is not None}
        return cls(**filtered)
