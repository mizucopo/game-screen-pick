"""model store client/serverのcanonical runtime identity。"""

import re
from dataclasses import dataclass

from .model_store_kind import ModelStoreKind

_SAFE_VERSION_PATTERN = re.compile(r"[0-9A-Za-z][0-9A-Za-z.+_-]{0,127}")


@dataclass(frozen=True)
class ModelRuntimeIdentity:
    """store kindとprivacy-safeなruntime versionの組。"""

    store_kind: ModelStoreKind
    version: str

    def __post_init__(self) -> None:
        """provenanceへ安全に記録できるversionだけを受け入れる。"""
        if _SAFE_VERSION_PATTERN.fullmatch(self.version) is None:
            msg = "安全なmodel runtime versionが必要です"
            raise ValueError(msg)

    @property
    def identifier(self) -> str:
        """store adapter間で衝突しないcanonical identityを返す。"""
        prefix = (
            "ollama" if self.store_kind is ModelStoreKind.OLLAMA else "huggingface-hub"
        )
        return f"{prefix}:{self.version}"
