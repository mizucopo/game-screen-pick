"""ModelRuntimeのsemantic port。"""

from typing import Protocol

from ..models.resolved_model_identity import ResolvedModelIdentity


class ModelRuntime(Protocol):
    """実行に使うmodel identityを解決する境界。"""

    def resolve_models(self) -> ResolvedModelIdentity:
        """freezeされたResolved Model Identityを返す。"""
