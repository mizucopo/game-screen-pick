"""一つのmodel roleのrun別解決結果。"""

from dataclasses import dataclass, field
from pathlib import Path

from .model_role import ModelRole
from .model_update_status import ModelUpdateStatus
from .resolved_model_identity import ResolvedModelIdentity


@dataclass(frozen=True)
class ResolvedModel:
    """設定名、更新前、更新結果、実行identityを分離して保持する。"""

    role: ModelRole
    configured_name: str
    canonical_name: str
    local_identity_before_update: ResolvedModelIdentity | None
    update_status: ModelUpdateStatus
    execution_identity: ResolvedModelIdentity
    runtime_identity: str
    artifact_location: Path | None = field(default=None, repr=False)

    def semantic_input(self) -> dict[str, object]:
        """model依存Stageのrole局所fingerprint入力を返す。"""
        return {
            "configured_name": self.configured_name,
            "execution_identity": self.execution_identity.identifier,
            "runtime_identity": self.runtime_identity,
            "store": self.execution_identity.store_kind.value,
        }

    def provenance(self) -> dict[str, object]:
        """pathやcredentialを含まないrun別provenanceを返す。"""
        return {
            "store": self.execution_identity.store_kind.value,
            "configured_name": self.configured_name,
            "canonical_name": self.canonical_name,
            "local_identity_before_update": (
                None
                if self.local_identity_before_update is None
                else self.local_identity_before_update.identifier
            ),
            "update_status": self.update_status.value,
            "execution_identity": self.execution_identity.identifier,
            "runtime_identity": self.runtime_identity,
        }
