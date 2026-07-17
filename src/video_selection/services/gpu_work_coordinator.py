"""同一process内のGPU-heavy work scheduling boundary。"""

from collections.abc import Callable
from threading import Lock
from typing import Literal, TypeVar

GpuWorkKind = Literal["speech_to_text", "vision_inference"]
WorkValue = TypeVar("WorkValue")


class GpuWorkCoordinator:
    """共有されたSTTとVision GPU workを直列実行する。"""

    def __init__(self) -> None:
        self._lock = Lock()

    def run(
        self,
        work_kind: GpuWorkKind,
        operation: Callable[[], WorkValue],
    ) -> WorkValue:
        """GPU leaseを一つだけ取得してoperationを実行する。"""
        if work_kind not in {"speech_to_text", "vision_inference"}:
            msg = "GPU work kindが不正です"
            raise ValueError(msg)
        with self._lock:
            return operation()
