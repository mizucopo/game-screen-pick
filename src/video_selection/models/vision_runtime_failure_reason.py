"""VisionRuntimeのstable failure reason。"""

from enum import StrEnum


class VisionRuntimeFailureReason(StrEnum):
    """retryと運用表示で使う失敗分類。"""

    TRANSPORT_FAILURE = "transport_failure"
    RESPONSE_INVALID = "response_invalid"
    SCHEMA_INVALID = "schema_invalid"
    DOMAIN_INVALID = "domain_invalid"
    MODEL_UNAVAILABLE = "model_unavailable"
    INVALID_REQUEST = "invalid_request"
