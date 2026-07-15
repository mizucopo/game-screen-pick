from src.video_selection.models.resolved_model_identity import ResolvedModelIdentity


class FakeModelRuntime:
    """固定されたResolved Model Identityを返すfake。"""

    def __init__(self, identity: ResolvedModelIdentity) -> None:
        self._identity = identity

    def resolve_models(self) -> ResolvedModelIdentity:
        """実行に使うmodel identityを返す。"""
        return self._identity
