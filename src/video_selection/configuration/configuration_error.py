"""動画入力設定のusage error。"""


class ConfigurationError(ValueError):
    """副作用前に返すexit 2相当の設定error。"""

    exit_code = 2

    def __init__(self, reason_code: str, message: str) -> None:
        """安全化済みmessageとreason codeを保持する。"""
        super().__init__(f"{reason_code}: {message}")
        self.reason_code = reason_code
