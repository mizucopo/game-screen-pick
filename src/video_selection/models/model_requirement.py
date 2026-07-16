"""一つのmodel roleへ要求するstore capability。"""

from dataclasses import dataclass

from .model_capability import ModelCapability
from .model_role import ModelRole
from .model_store_kind import ModelStoreKind


@dataclass(frozen=True)
class ModelRequirement:
    """Effective Configurationから導出されたmodel解決要求。"""

    role: ModelRole
    store_kind: ModelStoreKind
    configured_name: str
    capability: ModelCapability
    minimum_context_length: int | None = None
    device: str | None = None
    compute_type: str | None = None

    def __post_init__(self) -> None:
        """roleとcapability固有の必須値を検証する。"""
        if not self.configured_name.strip():
            msg = "model configured nameには空でない値が必要です"
            raise ValueError(msg)
        if self.capability is ModelCapability.VISION_STRUCTURED_OUTPUT:
            if (
                self.store_kind is not ModelStoreKind.OLLAMA
                or self.minimum_context_length is None
                or self.minimum_context_length < 1
                or self.device is not None
                or self.compute_type is not None
            ):
                msg = "vision model requirementが不正です"
                raise ValueError(msg)
            return
        if (
            self.role is not ModelRole.SPEECH_TO_TEXT
            or self.store_kind is not ModelStoreKind.HUGGING_FACE
            or not self.device
            or not self.compute_type
            or self.minimum_context_length is not None
        ):
            msg = "speech-to-text model requirementが不正です"
            raise ValueError(msg)
