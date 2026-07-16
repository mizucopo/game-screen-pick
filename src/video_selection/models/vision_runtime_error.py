"""VisionRuntimeの安全化済みfatal error。"""

from .vision_runtime_failure_reason import VisionRuntimeFailureReason


class VisionRuntimeError(RuntimeError):
    """外部detailを含めずstable reasonとvalidation codeを運ぶ。"""

    def __init__(
        self,
        reason: VisionRuntimeFailureReason,
        *,
        validation_code: str | None = None,
        attempt_count: int = 1,
        retry_after_seconds: float = 1.0,
    ) -> None:
        self.reason = reason
        self.validation_code = validation_code
        self.attempt_count = attempt_count
        self.retry_after_seconds = retry_after_seconds
        super().__init__(f"VisionRuntime failed: {reason.value}")
