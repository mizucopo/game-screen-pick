from pathlib import Path

import pytest

from src.video_selection.models.model_role import ModelRole
from src.video_selection.models.model_runtime_identity import ModelRuntimeIdentity
from src.video_selection.models.model_store_kind import ModelStoreKind
from src.video_selection.models.model_update_status import ModelUpdateStatus
from src.video_selection.models.resolved_model import ResolvedModel
from src.video_selection.models.resolved_model_identity import ResolvedModelIdentity
from src.video_selection.models.resolved_models import ResolvedModels


def test_all_role_resolutions_are_addressable_and_serialized_stably(
    tmp_path: Path,
) -> None:
    """全model roleがlookupされ安定したrole順でserializeされること。

    Arrange:
        - 3 roleのfreeze済みResolved Modelが用意される
    Act:
        - role lookup、semantic input、provenanceが取得される
    Assert:
        - 指定roleが返されrole keyが安定順に並ぶこと
    """
    # Arrange
    models = ResolvedModels(_all_models(tmp_path))

    # Act
    speech = models.for_role(ModelRole.SPEECH_TO_TEXT)
    semantic_input = models.semantic_input()
    provenance = models.provenance()

    # Assert
    assert speech.role is ModelRole.SPEECH_TO_TEXT
    assert tuple(semantic_input) == (
        "candidate_annotation",
        "scene_catalog",
        "speech_to_text",
    )
    assert tuple(provenance) == tuple(semantic_input)


def test_missing_or_duplicate_role_is_rejected(tmp_path: Path) -> None:
    """不足または重複するmodel role集合が拒否されること。

    Arrange:
        - 一つのroleが欠けたResolved Model集合が用意される
    Act:
        - Resolved Modelsの構築が試行される
    Assert:
        - 全role必須のvalidation errorになること
    """
    # Arrange
    incomplete = _all_models(tmp_path)[:2]

    # Act
    # Assert
    with pytest.raises(ValueError, match="全model role"):
        ResolvedModels(incomplete)


def test_unavailable_roles_are_returned_in_stable_order(tmp_path: Path) -> None:
    """更新不能のmodel roleだけが安定順で返されること。

    Arrange:
        - Scene CatalogとSTTがunavailableのResolved Modelsが用意される
    Act:
        - warning対象roleが取得される
    Assert:
        - unavailable roleだけがrole名順で返されること
    """
    # Arrange
    items = tuple(
        ResolvedModel(
            role=item.role,
            configured_name=item.configured_name,
            canonical_name=item.canonical_name,
            local_identity_before_update=item.local_identity_before_update,
            update_status=(
                ModelUpdateStatus.UNAVAILABLE
                if item.role in (ModelRole.SCENE_CATALOG, ModelRole.SPEECH_TO_TEXT)
                else item.update_status
            ),
            execution_identity=item.execution_identity,
            runtime_identity=item.runtime_identity,
            artifact_location=item.artifact_location,
        )
        for item in _all_models(tmp_path)
    )
    models = ResolvedModels(items)

    # Act
    roles = models.unavailable_roles()

    # Assert
    assert roles == (ModelRole.SCENE_CATALOG, ModelRole.SPEECH_TO_TEXT)


def _all_models(tmp_path: Path) -> tuple[ResolvedModel, ...]:
    vision_identity = ResolvedModelIdentity(
        ModelStoreKind.OLLAMA,
        "sha256:" + "a" * 64,
    )
    speech_identity = ResolvedModelIdentity(
        ModelStoreKind.HUGGING_FACE,
        "b" * 40,
    )
    vision_runtime = ModelRuntimeIdentity(ModelStoreKind.OLLAMA, "0.31.2")
    speech_runtime = ModelRuntimeIdentity(ModelStoreKind.HUGGING_FACE, "0.36.2")
    return (
        _model(
            ModelRole.SCENE_CATALOG,
            vision_identity,
            vision_runtime,
            None,
        ),
        _model(
            ModelRole.CANDIDATE_ANNOTATION,
            vision_identity,
            vision_runtime,
            None,
        ),
        _model(
            ModelRole.SPEECH_TO_TEXT,
            speech_identity,
            speech_runtime,
            tmp_path / ("b" * 40),
        ),
    )


def _model(
    role: ModelRole,
    identity: ResolvedModelIdentity,
    runtime_identity: ModelRuntimeIdentity,
    artifact_location: Path | None,
) -> ResolvedModel:
    return ResolvedModel(
        role=role,
        configured_name=f"configured/{role.value}",
        canonical_name=f"canonical/{role.value}",
        local_identity_before_update=identity,
        update_status=ModelUpdateStatus.NOT_REQUESTED,
        execution_identity=identity,
        runtime_identity=runtime_identity,
        artifact_location=artifact_location,
    )
