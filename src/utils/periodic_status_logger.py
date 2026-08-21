"""長時間処理中の生存状態を定期的にログ出力する."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from threading import Event, Thread
from typing import Protocol


class _WaitableStopEvent(Protocol):
    """定期ログの停止待ちに必要なevent interface."""

    def wait(self, timeout: float) -> bool:
        """timeout内に停止通知を受けたか返す."""
        ...


def _log_until_stopped(
    logger: logging.Logger,
    message: str,
    stop_event: _WaitableStopEvent,
    *,
    started_at: float,
    interval_seconds: float,
    clock: Callable[[], float],
) -> None:
    """停止通知を受けるまで一定間隔で動作状態を出力する."""
    while not stop_event.wait(interval_seconds):
        logger.info(
            "%s（開始から%.0f秒経過）",
            message,
            max(0.0, clock() - started_at),
        )


@contextmanager
def periodic_status_log(
    logger: logging.Logger,
    message: str,
    *,
    interval_seconds: float,
) -> Iterator[None]:
    """context内の処理が続く間、別threadから生存状態をログ出力する."""
    if interval_seconds <= 0:
        raise ValueError("定期状態ログの間隔は正の数で指定してください")
    stop_event = Event()
    started_at = time.monotonic()
    thread = Thread(
        target=_log_until_stopped,
        args=(logger, message, stop_event),
        kwargs={
            "started_at": started_at,
            "interval_seconds": interval_seconds,
            "clock": time.monotonic,
        },
        name="game-screen-pick-status-logger",
        daemon=True,
    )
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join()
