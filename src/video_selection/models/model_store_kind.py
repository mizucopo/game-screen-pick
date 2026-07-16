"""model artifact storeの種類。"""

from enum import StrEnum


class ModelStoreKind(StrEnum):
    """Resolved Model Identityの解決元を表す。"""

    OLLAMA = "ollama"
    HUGGING_FACE = "hugging_face"
