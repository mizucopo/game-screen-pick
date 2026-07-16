"""model lifecycleが利用するstore port。"""

from typing import Protocol

from ..models.model_artifact import ModelArtifact
from ..models.model_requirement import ModelRequirement
from ..models.model_store_kind import ModelStoreKind


class ModelStore(Protocol):
    """local解決、同期、capability検証をstore別に実装する。"""

    @property
    def kind(self) -> ModelStoreKind:
        """store kindを返す。"""

    def resolve_local(self, requirement: ModelRequirement) -> ModelArtifact | None:
        """完全性未検証のlocal artifact候補を返す。"""

    def synchronize(self, requirement: ModelRequirement) -> ModelArtifact:
        """remote selectorと同期したartifact候補を返す。"""

    def validate(
        self,
        artifact: ModelArtifact,
        requirement: ModelRequirement,
    ) -> None:
        """artifactの完全性とrole capabilityを検証する。"""
