"""storeから解決されたmodel artifact候補。"""

from dataclasses import dataclass, field
from pathlib import Path

from .model_runtime_identity import ModelRuntimeIdentity
from .model_store_kind import ModelStoreKind
from .resolved_model_identity import ResolvedModelIdentity


@dataclass(frozen=True)
class ModelArtifact:
    """identity、canonical名、runtime、非公開locationの組。"""

    identity: ResolvedModelIdentity
    canonical_name: str
    runtime_identity: ModelRuntimeIdentity
    location: Path | None = field(repr=False)

    def __post_init__(self) -> None:
        """store固有のartifact shapeを検証する。"""
        if not self.canonical_name.strip():
            msg = "model artifact metadataには空でない値が必要です"
            raise ValueError(msg)
        if self.runtime_identity.store_kind is not self.identity.store_kind:
            msg = "model artifact runtimeのstore kindが一致しません"
            raise ValueError(msg)
        if (
            self.identity.store_kind is ModelStoreKind.OLLAMA
            and self.location is not None
        ) or (
            self.identity.store_kind is ModelStoreKind.HUGGING_FACE
            and self.location is None
        ):
            msg = "model artifact locationがstore kindと一致しません"
            raise ValueError(msg)
