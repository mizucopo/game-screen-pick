"""ModelRuntimeのstable failure reason。"""

from enum import StrEnum


class ModelRuntimeFailureReason(StrEnum):
    """external error detailから独立したmodel failure分類。"""

    MODEL_STORE_UNAVAILABLE = "model_store_unavailable"
    MODEL_NOT_AVAILABLE = "model_not_available"
    MODEL_ARTIFACT_INVALID = "model_artifact_invalid"
