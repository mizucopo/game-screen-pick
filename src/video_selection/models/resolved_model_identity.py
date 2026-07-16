"""完全性を検証済みのResolved Model Identity。"""

import re
from dataclasses import dataclass
from typing import Self

from .model_store_kind import ModelStoreKind

_OLLAMA_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_HUGGING_FACE_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class ResolvedModelIdentity:
    """store kindと完全digestまたはcommit SHAの組。"""

    store_kind: ModelStoreKind
    value: str

    def __post_init__(self) -> None:
        """store固有の完全identityだけを受け入れる。"""
        pattern = (
            _OLLAMA_DIGEST_PATTERN
            if self.store_kind is ModelStoreKind.OLLAMA
            else _HUGGING_FACE_COMMIT_PATTERN
        )
        if pattern.fullmatch(self.value) is None:
            msg = "store contractを満たす完全なmodel identityが必要です"
            raise ValueError(msg)

    @property
    def identifier(self) -> str:
        """store間で衝突しないcanonical identityを返す。"""
        prefix = "ollama" if self.store_kind is ModelStoreKind.OLLAMA else "hf"
        return f"{prefix}:{self.value}"

    @classmethod
    def from_identifier(cls, identifier: str) -> Self:
        """canonical identifierを検証してtyped identityへ戻す。"""
        if identifier.startswith("ollama:"):
            return cls(ModelStoreKind.OLLAMA, identifier.removeprefix("ollama:"))
        if identifier.startswith("hf:"):
            return cls(ModelStoreKind.HUGGING_FACE, identifier.removeprefix("hf:"))
        raise ValueError("canonicalなmodel identity identifierが必要です")
