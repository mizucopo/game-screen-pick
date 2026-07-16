"""ModelRuntimeのsemantic port。"""

from typing import Protocol

from ..models.effective_configuration import EffectiveConfiguration
from ..models.resolved_models import ResolvedModels


class ModelRuntime(Protocol):
    """全roleのmodel lifecycleと実行identityを閉じ込める境界。"""

    def resolve_models(
        self,
        configuration: EffectiveConfiguration,
    ) -> ResolvedModels:
        """run内でfreezeされた全roleのResolved Modelを返す。"""
