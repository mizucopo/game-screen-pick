"""blocking external workのProgress Event watchdog。"""

from collections.abc import Callable
from threading import Event, Thread
from typing import TypeVar

from .run_progress_tracker import RunProgressTracker

WaitForStop = Callable[[Event, float], bool]
WorkValue = TypeVar("WorkValue")
_HEARTBEAT_INTERVAL_SECONDS = 30.0


class ExternalWorkMonitor:
    """external operation中に30秒間隔のheartbeatを発行する。"""

    def __init__(
        self,
        progress: RunProgressTracker,
        *,
        wait_for_stop: WaitForStop | None = None,
    ) -> None:
        self._progress = progress
        self._wait_for_stop = wait_for_stop or _wait_for_stop

    def run(
        self,
        operation: Callable[[], WorkValue],
        *,
        reason_code: str,
    ) -> WorkValue:
        """開始eventから完了までheartbeat watchdog付きで実行する。"""
        self._progress.external_work_started(reason_code)
        stop = Event()
        thread = Thread(
            target=self._emit_heartbeats,
            args=(stop,),
            daemon=True,
            name="progress-heartbeat",
        )
        thread.start()
        try:
            return operation()
        finally:
            stop.set()
            thread.join()

    def _emit_heartbeats(self, stop: Event) -> None:
        while not self._wait_for_stop(stop, _HEARTBEAT_INTERVAL_SECONDS):
            self._progress.heartbeat()


def _wait_for_stop(stop: Event, timeout_seconds: float) -> bool:
    return stop.wait(timeout_seconds)
