"""前回ログ行からの経過秒を付与するformatter."""

import logging


class ElapsedLogFormatter(logging.Formatter):
    """ログ行ごとに直前ログからの経過秒を出力するformatter."""

    def __init__(self, fmt: str, datefmt: str | None = None) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt)
        self._previous_created_at: float | None = None

    def format(self, record: logging.LogRecord) -> str:
        previous_created_at = self._previous_created_at
        elapsed_seconds = 0.0
        if previous_created_at is not None:
            elapsed_seconds = max(0.0, record.created - previous_created_at)
        self._previous_created_at = record.created
        record.elapsed_seconds = elapsed_seconds
        return super().format(record)
