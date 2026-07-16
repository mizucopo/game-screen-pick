import hashlib
from pathlib import Path

from src.video_selection.models.effective_configuration import EffectiveConfiguration
from src.video_selection.models.model_role import ModelRole
from src.video_selection.models.model_runtime_identity import ModelRuntimeIdentity
from src.video_selection.models.model_store_kind import ModelStoreKind
from src.video_selection.models.model_update_status import ModelUpdateStatus
from src.video_selection.models.resolved_model import ResolvedModel
from src.video_selection.models.resolved_model_identity import ResolvedModelIdentity
from src.video_selection.models.resolved_models import ResolvedModels


class FakeModelRuntime:
    """seedから固定された全roleのResolved Modelを返すfake。"""

    def __init__(
        self,
        candidate_identity_seed: str,
        *,
        speech_identity_seed: str = "speech-model",
        unavailable_roles: frozenset[ModelRole] = frozenset(),
    ) -> None:
        self._candidate_identity_seed = candidate_identity_seed
        self._speech_identity_seed = speech_identity_seed
        self._unavailable_roles = unavailable_roles

    def resolve_models(
        self,
        configuration: EffectiveConfiguration,
    ) -> ResolvedModels:
        """実行に使う全roleのmodel identityを返す。"""
        vision_identity = ResolvedModelIdentity(
            ModelStoreKind.OLLAMA,
            "sha256:"
            + hashlib.sha256(self._candidate_identity_seed.encode()).hexdigest(),
        )
        speech_identity = ResolvedModelIdentity(
            ModelStoreKind.HUGGING_FACE,
            hashlib.sha256(self._speech_identity_seed.encode()).hexdigest()[:40],
        )
        return ResolvedModels(
            (
                _resolved_model(
                    ModelRole.SCENE_CATALOG,
                    configuration.scene_catalog_model,
                    vision_identity,
                    self._unavailable_roles,
                ),
                _resolved_model(
                    ModelRole.CANDIDATE_ANNOTATION,
                    configuration.candidate_annotation_model,
                    vision_identity,
                    self._unavailable_roles,
                ),
                _resolved_model(
                    ModelRole.SPEECH_TO_TEXT,
                    configuration.speech_to_text_model,
                    speech_identity,
                    self._unavailable_roles,
                ),
            )
        )


def _resolved_model(
    role: ModelRole,
    configured_name: str,
    identity: ResolvedModelIdentity,
    unavailable_roles: frozenset[ModelRole],
) -> ResolvedModel:
    update_status = (
        ModelUpdateStatus.UNAVAILABLE
        if role in unavailable_roles
        else ModelUpdateStatus.NOT_REQUESTED
    )
    return ResolvedModel(
        role=role,
        configured_name=configured_name,
        canonical_name=configured_name,
        local_identity_before_update=identity,
        update_status=update_status,
        execution_identity=identity,
        runtime_identity=ModelRuntimeIdentity(identity.store_kind, "fake-1"),
        artifact_location=(
            Path("/fake/model")
            if identity.store_kind is ModelStoreKind.HUGGING_FACE
            else None
        ),
    )
