"""ModelRuntime境界のprivacy-safe error。"""

from .model_role import ModelRole
from .model_runtime_failure_reason import ModelRuntimeFailureReason


class ModelRuntimeError(RuntimeError):
    """安全な説明、stable reason、失敗roleを持つ。"""

    def __init__(
        self,
        reason: ModelRuntimeFailureReason,
        role: ModelRole,
    ) -> None:
        messages = {
            ModelRuntimeFailureReason.MODEL_STORE_UNAVAILABLE: (
                "model storeの状態を確認できませんでした"
            ),
            ModelRuntimeFailureReason.MODEL_NOT_AVAILABLE: (
                "実行可能なmodelを取得できませんでした"
            ),
            ModelRuntimeFailureReason.MODEL_ARTIFACT_INVALID: (
                "取得したmodelの完全性またはcapabilityを確認できませんでした"
            ),
        }
        super().__init__(f"{role.value}: {messages[reason]}")
        self.reason = reason
        self.role = role
