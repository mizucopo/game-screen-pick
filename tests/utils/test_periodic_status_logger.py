"""periodic_status_loggerの単体テスト."""

import logging
from collections.abc import Iterator
from unittest.mock import MagicMock, call

from src.utils.periodic_status_logger import _log_until_stopped


class FakeStopEvent:
    """指定したwait結果を順に返すevent fake."""

    def __init__(self, results: Iterator[bool]) -> None:
        self._results = results

    def wait(self, _timeout: float) -> bool:
        """次の停止状態を返す."""
        return next(self._results)


def test_log_until_stopped_reports_elapsed_time_periodically() -> None:
    """停止通知まで動作中の状態と経過秒を継続して出力すること."""
    logger = MagicMock(spec=logging.Logger)
    stop_event = FakeStopEvent(iter((False, False, True)))
    clock_values = iter((130.0, 160.0))

    _log_until_stopped(
        logger,
        "画像選定処理は動作中です",
        stop_event,
        started_at=100.0,
        interval_seconds=30.0,
        clock=lambda: next(clock_values),
    )

    assert logger.info.call_args_list == [
        call("%s（開始から%.0f秒経過）", "画像選定処理は動作中です", 30.0),
        call("%s（開始から%.0f秒経過）", "画像選定処理は動作中です", 60.0),
    ]
