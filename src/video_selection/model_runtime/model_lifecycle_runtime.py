"""model store差を閉じ込めるrun単位のlifecycle orchestration。"""

from collections.abc import Callable

from ..models.effective_configuration import EffectiveConfiguration
from ..models.model_artifact import ModelArtifact
from ..models.model_artifact_invalid_error import ModelArtifactInvalidError
from ..models.model_capability import ModelCapability
from ..models.model_requirement import ModelRequirement
from ..models.model_role import ModelRole
from ..models.model_runtime_error import ModelRuntimeError
from ..models.model_runtime_failure_reason import ModelRuntimeFailureReason
from ..models.model_store_kind import ModelStoreKind
from ..models.model_store_unavailable_error import ModelStoreUnavailableError
from ..models.model_update_status import ModelUpdateStatus
from ..models.resolved_model import ResolvedModel
from ..models.resolved_model_identity import ResolvedModelIdentity
from ..models.resolved_models import ResolvedModels
from ..protocols.model_store import ModelStore
from .hugging_face_model_store import HuggingFaceModelStore
from .ollama_model_store import OllamaModelStore

ModelStoreFactory = Callable[[EffectiveConfiguration], ModelStore]


class ModelLifecycleRuntime:
    """upgrade policyを適用し3 roleのmodelをrun内でfreezeする。"""

    def __init__(
        self,
        ollama_store_factory: ModelStoreFactory | None = None,
        hugging_face_store_factory: ModelStoreFactory | None = None,
    ) -> None:
        self._ollama_store_factory = ollama_store_factory or _build_ollama_store
        self._hugging_face_store_factory = (
            hugging_face_store_factory or _build_hugging_face_store
        )

    def resolve_models(
        self,
        configuration: EffectiveConfiguration,
    ) -> ResolvedModels:
        """distinct modelを一度ずつ解決しrole別provenanceを返す。"""
        stores = {
            ModelStoreKind.OLLAMA: self._ollama_store_factory(configuration),
            ModelStoreKind.HUGGING_FACE: self._hugging_face_store_factory(
                configuration
            ),
        }
        requirements_by_model: dict[
            tuple[ModelStoreKind, str],
            list[ModelRequirement],
        ] = {}
        for requirement in _build_requirements(configuration):
            requirements_by_model.setdefault(
                (requirement.store_kind, requirement.configured_name),
                [],
            ).append(requirement)

        resolutions: list[ResolvedModel] = []
        for (
            store_kind,
            _configured_name,
        ), requirements in requirements_by_model.items():
            store = stores[store_kind]
            if store.kind is not store_kind:
                msg = "ModelStore factoryが異なるstore kindを返しました"
                raise ValueError(msg)
            resolutions.extend(
                self._resolve_distinct_model(
                    store,
                    tuple(requirements),
                    auto_upgrade=configuration.models_auto_upgrade,
                )
            )
        return ResolvedModels(tuple(resolutions))

    def _resolve_distinct_model(
        self,
        store: ModelStore,
        requirements: tuple[ModelRequirement, ...],
        *,
        auto_upgrade: bool,
    ) -> tuple[ResolvedModel, ...]:
        """一つのconfigured modelを同期し全共有roleへ割り当てる。"""
        primary = requirements[0]
        try:
            local_candidate = store.resolve_local(primary)
        except ModelArtifactInvalidError:
            local_candidate = None
        except Exception:
            raise ModelRuntimeError(
                ModelRuntimeFailureReason.MODEL_STORE_UNAVAILABLE,
                primary.role,
            ) from None

        local = (
            local_candidate
            if local_candidate is not None
            and _artifact_is_valid(store, local_candidate, requirements)
            else None
        )
        if local is not None and not auto_upgrade:
            return _build_resolutions(
                requirements,
                local,
                local.identity,
                ModelUpdateStatus.NOT_REQUESTED,
            )

        try:
            synchronized = store.synchronize(primary)
        except ModelArtifactInvalidError:
            raise ModelRuntimeError(
                ModelRuntimeFailureReason.MODEL_ARTIFACT_INVALID,
                primary.role,
            ) from None
        except ModelStoreUnavailableError:
            if local is not None:
                fallback = _resolve_valid_local_after_unavailable(
                    store,
                    primary,
                    requirements,
                )
                if fallback is not None:
                    return _build_resolutions(
                        requirements,
                        fallback,
                        local.identity,
                        ModelUpdateStatus.UNAVAILABLE,
                    )
            raise ModelRuntimeError(
                ModelRuntimeFailureReason.MODEL_NOT_AVAILABLE,
                primary.role,
            ) from None
        except Exception:
            raise ModelRuntimeError(
                ModelRuntimeFailureReason.MODEL_NOT_AVAILABLE,
                primary.role,
            ) from None

        if not _artifact_is_valid(store, synchronized, requirements):
            raise ModelRuntimeError(
                ModelRuntimeFailureReason.MODEL_ARTIFACT_INVALID,
                primary.role,
            )
        if local is None:
            status = ModelUpdateStatus.BOOTSTRAPPED
        elif local.identity == synchronized.identity:
            status = ModelUpdateStatus.UNCHANGED
        else:
            status = ModelUpdateStatus.UPDATED
        return _build_resolutions(
            requirements,
            synchronized,
            None if local is None else local.identity,
            status,
        )


def _artifact_is_valid(
    store: ModelStore,
    artifact: ModelArtifact,
    requirements: tuple[ModelRequirement, ...],
) -> bool:
    if artifact.identity.store_kind is not store.kind:
        return False
    try:
        for requirement in requirements:
            store.validate(artifact, requirement)
    except Exception:
        return False
    return True


def _resolve_valid_local_after_unavailable(
    store: ModelStore,
    primary: ModelRequirement,
    requirements: tuple[ModelRequirement, ...],
) -> ModelArtifact | None:
    try:
        candidate = store.resolve_local(primary)
    except Exception:
        return None
    if candidate is None or not _artifact_is_valid(store, candidate, requirements):
        return None
    return candidate


def _build_resolutions(
    requirements: tuple[ModelRequirement, ...],
    artifact: ModelArtifact,
    local_identity_before_update: ResolvedModelIdentity | None,
    update_status: ModelUpdateStatus,
) -> tuple[ResolvedModel, ...]:
    return tuple(
        ResolvedModel(
            role=requirement.role,
            configured_name=requirement.configured_name,
            canonical_name=artifact.canonical_name,
            local_identity_before_update=local_identity_before_update,
            update_status=update_status,
            execution_identity=artifact.identity,
            runtime_identity=artifact.runtime_identity,
            artifact_location=artifact.location,
        )
        for requirement in requirements
    )


def _build_requirements(
    configuration: EffectiveConfiguration,
) -> tuple[ModelRequirement, ...]:
    return (
        ModelRequirement(
            role=ModelRole.SCENE_CATALOG,
            store_kind=ModelStoreKind.OLLAMA,
            configured_name=configuration.scene_catalog_model,
            capability=ModelCapability.VISION_STRUCTURED_OUTPUT,
            minimum_context_length=configuration.scene_catalog_num_ctx,
        ),
        ModelRequirement(
            role=ModelRole.CANDIDATE_ANNOTATION,
            store_kind=ModelStoreKind.OLLAMA,
            configured_name=configuration.candidate_annotation_model,
            capability=ModelCapability.VISION_STRUCTURED_OUTPUT,
            minimum_context_length=configuration.candidate_annotation_num_ctx,
        ),
        ModelRequirement(
            role=ModelRole.SPEECH_TO_TEXT,
            store_kind=ModelStoreKind.HUGGING_FACE,
            configured_name=configuration.speech_to_text_model,
            capability=ModelCapability.SPEECH_TO_TEXT,
            device=configuration.speech_to_text_device,
            compute_type=configuration.speech_to_text_compute_type,
        ),
    )


def _build_ollama_store(
    configuration: EffectiveConfiguration,
) -> ModelStore:
    return OllamaModelStore(
        configuration.ollama_host,
        timeout_seconds=configuration.ollama_timeout_seconds,
    )


def _build_hugging_face_store(
    _configuration: EffectiveConfiguration,
) -> ModelStore:
    return HuggingFaceModelStore()
