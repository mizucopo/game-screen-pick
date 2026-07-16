"""安全化済みのmodel store HTTP failure。"""

from .model_store_unavailable_error import ModelStoreUnavailableError


class ModelStoreHttpError(ModelStoreUnavailableError):
    """外部detailを捨て、statusと再試行待機だけを保持する。"""

    def __init__(
        self,
        status_code: int,
        *,
        retry_after_seconds: float = 1.0,
    ) -> None:
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds
        super().__init__("Model store HTTP request failed")
