"""elapsed_log_formatterの単体テスト."""

import logging
from datetime import datetime

from src.utils.elapsed_log_formatter import ElapsedLogFormatter


def test_format_adds_timestamp_and_elapsed_seconds() -> None:
    """ログ行の先頭に日時と前回ログからの経過秒が付与されること.

    Arrange:
        - 直前ログから2.5秒後のログrecordがある
    Act:
        - formatterでログ行が整形される
    Assert:
        - 日時と経過秒がログ行の先頭に出力されること
    """
    # Arrange
    formatter = ElapsedLogFormatter(
        fmt="%(asctime)s.%(msecs)03d +%(elapsed_seconds).3fs %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    first_record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="開始",
        args=(),
        exc_info=None,
    )
    second_record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=2,
        msg="完了",
        args=(),
        exc_info=None,
    )
    started_at = datetime(2026, 8, 6, 13, 0, 0)
    first_record.created = started_at.timestamp() + 0.123
    first_record.msecs = 123.0
    second_record.created = started_at.timestamp() + 2.623
    second_record.msecs = 623.0

    # Act
    first_line = formatter.format(first_record)
    second_line = formatter.format(second_record)

    # Assert
    assert first_line == "2026-08-06 13:00:00.123 +0.000s 開始"
    assert second_line == "2026-08-06 13:00:02.623 +2.500s 完了"
