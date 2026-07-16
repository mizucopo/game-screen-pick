"""一つのmodel roleのrun別解決結果。"""

from dataclasses import dataclass, field
from pathlib import Path

from .model_role import ModelRole
from .model_runtime_identity import ModelRuntimeIdentity
from .model_store_kind import ModelStoreKind
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
    runtime_identity: ModelRuntimeIdentity
    artifact_location: Path | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """role、store、更新状態のrun別contractを検証する。"""
        if not self.configured_name.strip() or not self.canonical_name.strip():
            msg = "Resolved Modelのmodel名には空でない値が必要です"
            raise ValueError(msg)
        expected_store = (
            ModelStoreKind.HUGGING_FACE
            if self.role is ModelRole.SPEECH_TO_TEXT
            else ModelStoreKind.OLLAMA
        )
        identities_match_store = (
            self.execution_identity.store_kind is expected_store
            and self.runtime_identity.store_kind is expected_store
            and (
                self.local_identity_before_update is None
                or self.local_identity_before_update.store_kind is expected_store
            )
        )
        if not identities_match_store:
            msg = "Resolved Modelのroleとstore kindが一致しません"
            raise ValueError(msg)
        if (expected_store is ModelStoreKind.OLLAMA) == (
            self.artifact_location is not None
        ):
            msg = "Resolved Modelのartifact locationがstore kindと一致しません"
            raise ValueError(msg)
        self._validate_update_state()

    def _validate_update_state(self) -> None:
        """更新前identity、status、実行identityの整合を検証する。"""
        before = self.local_identity_before_update
        if self.update_status is ModelUpdateStatus.BOOTSTRAPPED:
            valid = before is None
        elif self.update_status is ModelUpdateStatus.UPDATED:
            valid = before is not None and before != self.execution_identity
        else:
            valid = before == self.execution_identity
        if not valid:
            msg = "Resolved Modelのupdate stateが一致しません"
            raise ValueError(msg)

    def semantic_input(self) -> dict[str, object]:
        """model依存Stageのrole局所fingerprint入力を返す。"""
        return {
            "configured_name": self.configured_name,
            "execution_identity": self.execution_identity.identifier,
            "runtime_identity": self.runtime_identity.identifier,
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
            "runtime_identity": self.runtime_identity.identifier,
        }
