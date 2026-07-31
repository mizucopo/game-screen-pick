"""Acceptance Run間で同じResolved Model identityを返すruntime。"""

from ..models.effective_configuration import EffectiveConfiguration
from ..models.resolved_models import ResolvedModels


class FrozenModelRuntime:
    """事前解決済みmodel集合をrun内で再解決せず返す。"""

    def __init__(self, models: ResolvedModels) -> None:
        self._models = models

    def resolve_models(
        self,
        configuration: EffectiveConfiguration,
    ) -> ResolvedModels:
        """configurationのmodel selectorがfreeze時と一致する場合だけ返す。"""
        configured = {
            "scene_catalog": configuration.scene_catalog_model,
            "candidate_annotation": configuration.candidate_annotation_model,
            "speech_to_text": configuration.speech_to_text_model,
        }
        actual = {item.role.value: item.configured_name for item in self._models.items}
        if configured != actual:
            raise ValueError("Acceptance Runのmodel設定がfreeze時から変化しました")
        return self._models
