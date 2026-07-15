"""MediaRuntime境界のreason-coded failure。"""

from .media_runtime_failure_reason import MediaRuntimeFailureReason


class MediaRuntimeError(RuntimeError):
    """安全な説明とstable reasonを持つmedia operation error。"""

    def __init__(
        self,
        reason: MediaRuntimeFailureReason,
        message: str,
    ) -> None:
        super().__init__(message)
        self.reason = reason
