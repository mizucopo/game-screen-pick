"""同一process内のGPU-heavy work scheduling boundary。"""

from collections.abc import Callable
from threading import Condition
from typing import Literal, TypeVar

GpuWorkKind = Literal["speech_to_text", "vision_inference"]
WorkValue = TypeVar("WorkValue")


class GpuWorkCoordinator:
    """STTを排他にしVisionだけを設定上限まで並列実行する。"""

    def __init__(self, *, max_parallel_requests: int = 1) -> None:
        if (
            not isinstance(max_parallel_requests, int)
            or isinstance(max_parallel_requests, bool)
            or max_parallel_requests < 1
        ):
            msg = "Vision同時実行上限には1以上の整数が必要です"
            raise ValueError(msg)
        self._condition = Condition()
        self._max_parallel_requests = max_parallel_requests
        self._active_vision_count = 0
        self._speech_active = False
        self._speech_waiter_count = 0

    def run(
        self,
        work_kind: GpuWorkKind,
        operation: Callable[[], WorkValue],
    ) -> WorkValue:
        """work kindに応じたGPU leaseを取得してoperationを実行する。"""
        if work_kind not in {"speech_to_text", "vision_inference"}:
            msg = "GPU work kindが不正です"
            raise ValueError(msg)
        if work_kind == "speech_to_text":
            return self._run_speech(operation)
        return self._run_vision(operation)

    def _run_speech(self, operation: Callable[[], WorkValue]) -> WorkValue:
        """既存Visionの完了を待ちSTTを単独実行する。"""
        with self._condition:
            self._speech_waiter_count += 1
            try:
                while self._speech_active or self._active_vision_count:
                    self._condition.wait()
                self._speech_active = True
            finally:
                self._speech_waiter_count -= 1
        try:
            return operation()
        finally:
            with self._condition:
                self._speech_active = False
                self._condition.notify_all()

    def _run_vision(self, operation: Callable[[], WorkValue]) -> WorkValue:
        """STTがない間だけ設定上限までVisionを実行する。"""
        with self._condition:
            while (
                self._speech_active
                or self._speech_waiter_count
                or self._active_vision_count >= self._max_parallel_requests
            ):
                self._condition.wait()
            self._active_vision_count += 1
        try:
            return operation()
        finally:
            with self._condition:
                self._active_vision_count -= 1
                self._condition.notify_all()
