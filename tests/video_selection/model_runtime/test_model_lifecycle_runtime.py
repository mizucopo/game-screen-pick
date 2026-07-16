import json
import traceback
from pathlib import Path

import pytest

from src.video_selection.model_runtime.model_lifecycle_runtime import (
    ModelLifecycleRuntime,
)
from src.video_selection.models.effective_configuration import EffectiveConfiguration
from src.video_selection.models.model_artifact import ModelArtifact
from src.video_selection.models.model_artifact_invalid_error import (
    ModelArtifactInvalidError,
)
from src.video_selection.models.model_role import ModelRole
from src.video_selection.models.model_runtime_error import ModelRuntimeError
from src.video_selection.models.model_runtime_failure_reason import (
    ModelRuntimeFailureReason,
)
from src.video_selection.models.model_runtime_identity import ModelRuntimeIdentity
from src.video_selection.models.model_store_kind import ModelStoreKind
from src.video_selection.models.model_store_unavailable_error import (
    ModelStoreUnavailableError,
)
from src.video_selection.models.model_update_status import ModelUpdateStatus
from src.video_selection.models.resolved_model_identity import ResolvedModelIdentity
from src.video_selection.models.resolved_models import ResolvedModels
from tests.video_selection.fakes.fake_model_store import FakeModelStore


def test_updates_each_distinct_model_once_and_freezes_role_resolutions(
    tmp_path: Path,
) -> None:
    """distinct modelが一度ずつ更新されrole別の実行identityへfreezeされること。

    Arrange:
        - 2 roleで共有されるOllama tagと一つのSTT repoが用意される
        - 各storeに更新前と更新後の完全artifactが用意される
    Act:
        - auto upgrade有効で全modelが解決される
    Assert:
        - 共有tagが一度だけ同期され、3 roleのprovenanceが分離されること
    """
    # Arrange
    configuration = _configuration(tmp_path, auto_upgrade=True)
    old_ollama = _ollama_artifact("1")
    new_ollama = _ollama_artifact("2")
    old_hugging_face = _hugging_face_artifact(tmp_path, "3")
    new_hugging_face = _hugging_face_artifact(tmp_path, "4")
    ollama_store = FakeModelStore(
        ModelStoreKind.OLLAMA,
        local_artifacts={configuration.scene_catalog_model: old_ollama},
        synchronized_artifacts={configuration.scene_catalog_model: new_ollama},
    )
    hugging_face_store = FakeModelStore(
        ModelStoreKind.HUGGING_FACE,
        local_artifacts={configuration.speech_to_text_model: old_hugging_face},
        synchronized_artifacts={configuration.speech_to_text_model: new_hugging_face},
    )
    runtime = _runtime(ollama_store, hugging_face_store)

    # Act
    resolved = runtime.resolve_models(configuration)

    # Assert
    assert ollama_store.local_resolution_calls == [configuration.scene_catalog_model]
    assert ollama_store.synchronization_calls == [configuration.scene_catalog_model]
    assert ollama_store.publication_calls == [new_ollama.identity.identifier]
    assert ollama_store.identity_confirmation_calls == [
        f"{configuration.scene_catalog_model}:{old_ollama.identity.identifier}",
        f"{configuration.scene_catalog_model}:{new_ollama.identity.identifier}",
    ]
    assert hugging_face_store.local_resolution_calls == [
        configuration.speech_to_text_model
    ]
    assert hugging_face_store.synchronization_calls == [
        configuration.speech_to_text_model
    ]
    assert hugging_face_store.publication_calls == [
        new_hugging_face.identity.identifier
    ]
    assert hugging_face_store.identity_confirmation_calls == [
        (
            f"{configuration.speech_to_text_model}:"
            f"{old_hugging_face.identity.identifier}"
        ),
        (
            f"{configuration.speech_to_text_model}:"
            f"{new_hugging_face.identity.identifier}"
        ),
    ]
    for role in (ModelRole.SCENE_CATALOG, ModelRole.CANDIDATE_ANNOTATION):
        resolution = resolved.for_role(role)
        assert resolution.local_identity_before_update == old_ollama.identity
        assert resolution.update_status is ModelUpdateStatus.UPDATED
        assert resolution.execution_identity == new_ollama.identity
    speech = resolved.for_role(ModelRole.SPEECH_TO_TEXT)
    assert speech.local_identity_before_update == old_hugging_face.identity
    assert speech.update_status is ModelUpdateStatus.UPDATED
    assert speech.execution_identity == new_hugging_face.identity


def test_equivalent_ollama_latest_names_are_synchronized_once(tmp_path: Path) -> None:
    """省略tagとlatest表記が同じOllama modelとして一度だけ同期されること。

    Arrange:
        - Scene Catalogに省略tag、Candidate Annotationにlatest表記が用意される
        - 両表記が指す一つのlocal・同期後artifactが用意される
    Act:
        - auto upgrade有効で全modelが解決される
    Assert:
        - equivalent tagが一度だけlocal解決・同期され両roleへfreezeされること
    """
    # Arrange
    configuration = EffectiveConfiguration(
        video_input_folder=tmp_path / "videos",
        output_folder=tmp_path / "output",
        scene_catalog_model="qwen3-vl",
        candidate_annotation_model="qwen3-vl:latest",
        models_auto_upgrade=True,
    )
    old_ollama = _ollama_artifact("1")
    new_ollama = _ollama_artifact("2")
    hugging_face = _hugging_face_artifact(tmp_path, "3")
    ollama_store = FakeModelStore(
        ModelStoreKind.OLLAMA,
        local_artifacts={configuration.scene_catalog_model: old_ollama},
        synchronized_artifacts={configuration.scene_catalog_model: new_ollama},
    )
    hugging_face_store = FakeModelStore(
        ModelStoreKind.HUGGING_FACE,
        local_artifacts={configuration.speech_to_text_model: hugging_face},
        synchronized_artifacts={configuration.speech_to_text_model: hugging_face},
    )

    # Act
    resolved = _runtime(ollama_store, hugging_face_store).resolve_models(configuration)

    # Assert
    assert ollama_store.local_resolution_calls == ["qwen3-vl"]
    assert ollama_store.synchronization_calls == ["qwen3-vl"]
    assert resolved.for_role(ModelRole.SCENE_CATALOG).execution_identity == (
        new_ollama.identity
    )
    assert resolved.for_role(ModelRole.CANDIDATE_ANNOTATION).execution_identity == (
        new_ollama.identity
    )


def test_same_execution_identity_preserves_semantic_fingerprint_input(
    tmp_path: Path,
) -> None:
    """更新結果が異なっても同じ実行identityのsemantic inputが維持されること。

    Arrange:
        - 同じlocal artifactに対する更新成功と更新不能の2 runが用意される
    Act:
        - 両runのcandidate model resolutionが解決される
    Assert:
        - provenanceは異なりsemantic fingerprint inputは一致すること
    """
    # Arrange
    configuration = _configuration(tmp_path, auto_upgrade=True)
    ollama = _ollama_artifact("1")
    hugging_face = _hugging_face_artifact(tmp_path, "2")
    successful = _runtime(
        FakeModelStore(
            ModelStoreKind.OLLAMA,
            local_artifacts={configuration.scene_catalog_model: ollama},
            synchronized_artifacts={configuration.scene_catalog_model: ollama},
        ),
        FakeModelStore(
            ModelStoreKind.HUGGING_FACE,
            local_artifacts={configuration.speech_to_text_model: hugging_face},
            synchronized_artifacts={configuration.speech_to_text_model: hugging_face},
        ),
    )
    unavailable = _runtime(
        FakeModelStore(
            ModelStoreKind.OLLAMA,
            local_artifacts={configuration.scene_catalog_model: ollama},
            synchronization_errors={
                configuration.scene_catalog_model: ModelStoreUnavailableError(
                    "registry detail"
                )
            },
        ),
        FakeModelStore(
            ModelStoreKind.HUGGING_FACE,
            local_artifacts={configuration.speech_to_text_model: hugging_face},
            synchronization_errors={
                configuration.speech_to_text_model: ModelStoreUnavailableError(
                    "hub detail"
                )
            },
        ),
    )

    # Act
    same = successful.resolve_models(configuration).for_role(
        ModelRole.CANDIDATE_ANNOTATION
    )
    offline = unavailable.resolve_models(configuration).for_role(
        ModelRole.CANDIDATE_ANNOTATION
    )

    # Assert
    assert same.update_status is ModelUpdateStatus.UNCHANGED
    assert offline.update_status is ModelUpdateStatus.UNAVAILABLE
    assert same.provenance() != offline.provenance()
    assert same.semantic_input() == offline.semantic_input()


def test_auto_upgrade_false_uses_complete_local_models_without_network(
    tmp_path: Path,
) -> None:
    """auto upgrade無効時に完全なlocal modelが同期なしで使用されること。

    Arrange:
        - 3 roleに必要な完全local artifactが用意される
        - auto upgradeが無効にされる
    Act:
        - 全modelが解決される
    Assert:
        - network同期が行われずnot_requestedとしてfreezeされること
    """
    # Arrange
    configuration = _configuration(tmp_path, auto_upgrade=False)
    ollama = _ollama_artifact("1")
    hugging_face = _hugging_face_artifact(tmp_path, "2")
    ollama_store = FakeModelStore(
        ModelStoreKind.OLLAMA,
        local_artifacts={configuration.scene_catalog_model: ollama},
    )
    hugging_face_store = FakeModelStore(
        ModelStoreKind.HUGGING_FACE,
        local_artifacts={configuration.speech_to_text_model: hugging_face},
    )

    # Act
    resolved = _runtime(ollama_store, hugging_face_store).resolve_models(configuration)

    # Assert
    assert ollama_store.synchronization_calls == []
    assert hugging_face_store.synchronization_calls == []
    assert all(
        item.update_status is ModelUpdateStatus.NOT_REQUESTED for item in resolved.items
    )


def test_missing_local_model_is_bootstrapped_when_auto_upgrade_is_false(
    tmp_path: Path,
) -> None:
    """auto upgrade無効でもmissing local modelがbootstrapされること。

    Arrange:
        - local storeにmodelがなく同期可能なartifactが用意される
        - auto upgradeが無効にされる
    Act:
        - 全modelが解決される
    Assert:
        - distinct modelが一度ずつ取得されbootstrappedと記録されること
    """
    # Arrange
    configuration = _configuration(tmp_path, auto_upgrade=False)
    ollama = _ollama_artifact("1")
    hugging_face = _hugging_face_artifact(tmp_path, "2")
    ollama_store = FakeModelStore(
        ModelStoreKind.OLLAMA,
        local_artifacts={configuration.scene_catalog_model: None},
        synchronized_artifacts={configuration.scene_catalog_model: ollama},
    )
    hugging_face_store = FakeModelStore(
        ModelStoreKind.HUGGING_FACE,
        local_artifacts={configuration.speech_to_text_model: None},
        synchronized_artifacts={configuration.speech_to_text_model: hugging_face},
    )

    # Act
    resolved = _runtime(ollama_store, hugging_face_store).resolve_models(configuration)

    # Assert
    assert ollama_store.synchronization_calls == [configuration.scene_catalog_model]
    assert hugging_face_store.synchronization_calls == [
        configuration.speech_to_text_model
    ]
    assert all(
        item.local_identity_before_update is None
        and item.update_status is ModelUpdateStatus.BOOTSTRAPPED
        for item in resolved.items
    )


def test_unavailable_update_uses_only_a_validated_local_artifact(
    tmp_path: Path,
) -> None:
    """更新不能時に検証済みlocal artifactだけがwarning付きで使用されること。

    Arrange:
        - 完全なlocal artifactと同期不能errorが各storeに用意される
    Act:
        - auto upgrade有効で全modelが解決される
    Assert:
        - local identityが実行identityとなりunavailableが記録されること
    """
    # Arrange
    configuration = _configuration(tmp_path, auto_upgrade=True)
    ollama = _ollama_artifact("1")
    hugging_face = _hugging_face_artifact(tmp_path, "2")
    runtime = _runtime(
        FakeModelStore(
            ModelStoreKind.OLLAMA,
            local_artifacts={configuration.scene_catalog_model: ollama},
            synchronization_errors={
                configuration.scene_catalog_model: ModelStoreUnavailableError("offline")
            },
        ),
        FakeModelStore(
            ModelStoreKind.HUGGING_FACE,
            local_artifacts={configuration.speech_to_text_model: hugging_face},
            synchronization_errors={
                configuration.speech_to_text_model: ModelStoreUnavailableError(
                    "offline"
                )
            },
        ),
    )

    # Act
    resolved = runtime.resolve_models(configuration)

    # Assert
    assert all(
        item.update_status is ModelUpdateStatus.UNAVAILABLE
        and item.local_identity_before_update == item.execution_identity
        for item in resolved.items
    )


def test_validated_artifact_publication_failure_reuses_old_local_artifact(
    tmp_path: Path,
) -> None:
    """検証後のlocal selector公開失敗で更新前artifactが再検査されること。

    Arrange:
        - 完全な更新前artifactと検証可能な同期後artifactが用意される
        - 同期後artifactのlocal selector公開だけが利用不能にされる
    Act:
        - auto upgrade有効でmodel解決が実行される
    Assert:
        - 更新後identityをfreezeせず再検査済み更新前identityが使われること
        - update statusがunavailableになること
    """
    # Arrange
    configuration = _configuration(tmp_path, auto_upgrade=True)
    old_ollama = _ollama_artifact("1")
    new_ollama = _ollama_artifact("2")
    hugging_face = _hugging_face_artifact(tmp_path, "3")
    ollama_store = FakeModelStore(
        ModelStoreKind.OLLAMA,
        local_artifacts={configuration.scene_catalog_model: old_ollama},
        synchronized_artifacts={configuration.scene_catalog_model: new_ollama},
        publication_errors={
            new_ollama.identity.identifier: ModelStoreUnavailableError(
                "local ref unavailable"
            )
        },
    )
    hugging_face_store = FakeModelStore(
        ModelStoreKind.HUGGING_FACE,
        local_artifacts={configuration.speech_to_text_model: hugging_face},
        synchronized_artifacts={configuration.speech_to_text_model: hugging_face},
    )

    # Act
    resolved = _runtime(ollama_store, hugging_face_store).resolve_models(configuration)

    # Assert
    candidate = resolved.for_role(ModelRole.CANDIDATE_ANNOTATION)
    assert candidate.execution_identity == old_ollama.identity
    assert candidate.update_status is ModelUpdateStatus.UNAVAILABLE
    assert ollama_store.publication_calls == [new_ollama.identity.identifier]


def test_unavailable_update_rechecks_local_store_and_rejects_partial_state(
    tmp_path: Path,
) -> None:
    """更新不能後にlocal storeが再検査されpartial stateが拒否されること。

    Arrange:
        - 更新前には完全なlocal artifactが用意される
        - 同期failure後にlocal tagがpartial artifactへ変わるfakeが用意される
    Act:
        - auto upgrade有効でmodel解決が試行される
    Assert:
        - 更新前の観測だけを根拠に継続せずfatalになること
    """
    # Arrange
    configuration = _configuration(tmp_path, auto_upgrade=True)
    old = _ollama_artifact("1")
    partial = _ollama_artifact("2")
    runtime = _runtime(
        FakeModelStore(
            ModelStoreKind.OLLAMA,
            local_artifacts={configuration.scene_catalog_model: old},
            synchronization_errors={
                configuration.scene_catalog_model: ModelStoreUnavailableError(
                    "registry unavailable"
                )
            },
            local_artifacts_after_synchronization_error={
                configuration.scene_catalog_model: partial
            },
            invalid_identifiers=frozenset({partial.identity.identifier}),
        ),
        FakeModelStore(
            ModelStoreKind.HUGGING_FACE,
            local_artifacts={configuration.speech_to_text_model: None},
        ),
    )

    # Act
    # Assert
    with pytest.raises(ModelRuntimeError) as captured:
        runtime.resolve_models(configuration)
    assert captured.value.reason is ModelRuntimeFailureReason.MODEL_NOT_AVAILABLE
    assert captured.value.role is ModelRole.SCENE_CATALOG


def test_missing_or_partial_local_model_cannot_mask_bootstrap_failure(
    tmp_path: Path,
) -> None:
    """missingまたはpartial local modelでbootstrap failureがfatalになること。

    Arrange:
        - capability検証に失敗するlocal artifactが用意される
        - bootstrap同期も利用不能にされる
    Act:
        - model解決が試行される
    Assert:
        - partial artifactへfallbackせず安全なfatal errorになること
    """
    # Arrange
    configuration = _configuration(tmp_path, auto_upgrade=True)
    partial = _ollama_artifact("1")
    ollama_store = FakeModelStore(
        ModelStoreKind.OLLAMA,
        local_artifacts={configuration.scene_catalog_model: partial},
        synchronization_errors={
            configuration.scene_catalog_model: ModelStoreUnavailableError(
                "/private/model-store/token-secret"
            )
        },
        invalid_identifiers=frozenset({partial.identity.identifier}),
    )
    hugging_face_store = FakeModelStore(
        ModelStoreKind.HUGGING_FACE,
        local_artifacts={configuration.speech_to_text_model: None},
    )
    runtime = _runtime(ollama_store, hugging_face_store)

    # Act
    # Assert
    with pytest.raises(ModelRuntimeError) as captured:
        runtime.resolve_models(configuration)
    assert captured.value.reason is ModelRuntimeFailureReason.MODEL_NOT_AVAILABLE
    assert "/private/model-store" not in str(captured.value)
    assert "token-secret" not in str(captured.value)
    formatted = "".join(
        traceback.format_exception(captured.type, captured.value, captured.tb)
    )
    assert "/private/model-store" not in formatted
    assert "token-secret" not in formatted


def test_invalid_synchronized_artifact_is_fatal_without_old_model_fallback(
    tmp_path: Path,
) -> None:
    """同期後artifactがpartialなら更新前modelへfallbackせず失敗すること。

    Arrange:
        - 完全な更新前artifactとcapability検証に失敗する同期後artifactがある
    Act:
        - auto upgrade有効でmodel解決が試行される
    Assert:
        - 更新前artifactを実行identityにせずfatal errorになること
    """
    # Arrange
    configuration = _configuration(tmp_path, auto_upgrade=True)
    old = _ollama_artifact("1")
    partial = _ollama_artifact("2")
    runtime = _runtime(
        FakeModelStore(
            ModelStoreKind.OLLAMA,
            local_artifacts={configuration.scene_catalog_model: old},
            synchronized_artifacts={configuration.scene_catalog_model: partial},
            invalid_identifiers=frozenset({partial.identity.identifier}),
        ),
        FakeModelStore(
            ModelStoreKind.HUGGING_FACE,
            local_artifacts={configuration.speech_to_text_model: None},
        ),
    )

    # Act
    # Assert
    with pytest.raises(ModelRuntimeError) as captured:
        runtime.resolve_models(configuration)
    assert captured.value.reason is ModelRuntimeFailureReason.MODEL_ARTIFACT_INVALID


def test_shared_model_capability_failure_reports_rejecting_role(
    tmp_path: Path,
) -> None:
    """共有modelのcapability失敗が検査を拒否したroleへ帰属されること。

    Arrange:
        - 2 roleが共有する同期後Ollama artifactが用意される
        - Candidate Annotation roleだけのcapability検証が失敗するようにされる
    Act:
        - auto upgrade有効でmodel解決が試行される
    Assert:
        - artifact invalid errorのroleがCandidate Annotationになること
    """
    # Arrange
    configuration = _configuration(tmp_path, auto_upgrade=True)
    synchronized = _ollama_artifact("2")
    runtime = _runtime(
        FakeModelStore(
            ModelStoreKind.OLLAMA,
            local_artifacts={configuration.scene_catalog_model: None},
            synchronized_artifacts={configuration.scene_catalog_model: synchronized},
            invalid_artifact_roles=frozenset(
                {
                    (
                        synchronized.identity.identifier,
                        ModelRole.CANDIDATE_ANNOTATION,
                    )
                }
            ),
        ),
        FakeModelStore(
            ModelStoreKind.HUGGING_FACE,
            local_artifacts={configuration.speech_to_text_model: None},
        ),
    )

    # Act
    # Assert
    with pytest.raises(ModelRuntimeError) as captured:
        runtime.resolve_models(configuration)
    assert captured.value.reason is ModelRuntimeFailureReason.MODEL_ARTIFACT_INVALID
    assert captured.value.role is ModelRole.CANDIDATE_ANNOTATION


def test_unavailable_sync_preserves_role_that_rejected_local_shared_model(
    tmp_path: Path,
) -> None:
    """共有local検証後の同期不能が最初に拒否したroleへ帰属されること。

    Arrange:
        - 2 roleが共有するlocal Ollama artifactが用意される
        - Candidate Annotationだけがlocalを拒否しremote同期も不能にされる
    Act:
        - auto upgrade有効でmodel解決が試行される
    Assert:
        - model not available errorのroleがCandidate Annotationになること
    """
    # Arrange
    configuration = _configuration(tmp_path, auto_upgrade=True)
    local = _ollama_artifact("1")
    runtime = _runtime(
        FakeModelStore(
            ModelStoreKind.OLLAMA,
            local_artifacts={configuration.scene_catalog_model: local},
            synchronization_errors={
                configuration.scene_catalog_model: ModelStoreUnavailableError("offline")
            },
            invalid_artifact_roles=frozenset(
                {
                    (
                        local.identity.identifier,
                        ModelRole.CANDIDATE_ANNOTATION,
                    )
                }
            ),
        ),
        FakeModelStore(
            ModelStoreKind.HUGGING_FACE,
            local_artifacts={configuration.speech_to_text_model: None},
        ),
    )

    # Act
    # Assert
    with pytest.raises(ModelRuntimeError) as captured:
        runtime.resolve_models(configuration)
    assert captured.value.reason is ModelRuntimeFailureReason.MODEL_NOT_AVAILABLE
    assert captured.value.role is ModelRole.CANDIDATE_ANNOTATION


def test_invalid_synchronization_result_cannot_be_treated_as_unavailable(
    tmp_path: Path,
) -> None:
    """同期処理がpartialを報告した場合にoffline fallbackされないこと。

    Arrange:
        - 完全な更新前artifactが用意される
        - store同期がartifact invalid errorを返すようにされる
    Act:
        - auto upgrade有効でmodel解決が試行される
    Assert:
        - unavailable扱いで更新前modelを使わずfatalになること
    """
    # Arrange
    configuration = _configuration(tmp_path, auto_upgrade=True)
    old = _ollama_artifact("1")
    runtime = _runtime(
        FakeModelStore(
            ModelStoreKind.OLLAMA,
            local_artifacts={configuration.scene_catalog_model: old},
            synchronization_errors={
                configuration.scene_catalog_model: ModelArtifactInvalidError(
                    "partial download detail"
                )
            },
        ),
        FakeModelStore(
            ModelStoreKind.HUGGING_FACE,
            local_artifacts={configuration.speech_to_text_model: None},
        ),
    )

    # Act
    # Assert
    with pytest.raises(ModelRuntimeError) as captured:
        runtime.resolve_models(configuration)
    assert captured.value.reason is ModelRuntimeFailureReason.MODEL_ARTIFACT_INVALID


def test_model_role_identity_changes_only_its_semantic_input(
    tmp_path: Path,
) -> None:
    """一つのmodel role変更が無関係なroleのsemantic inputを変えないこと。

    Arrange:
        - Ollama identityが同じでSTT identityだけが異なる2 runが用意される
    Act:
        - 両runのrole別semantic inputが取得される
    Assert:
        - candidate roleは一致しSTT roleだけが変わること
    """
    # Arrange
    configuration = _configuration(tmp_path, auto_upgrade=False)
    ollama = _ollama_artifact("1")

    def resolve_with_speech(fill: str) -> ResolvedModels:
        return _runtime(
            FakeModelStore(
                ModelStoreKind.OLLAMA,
                local_artifacts={configuration.scene_catalog_model: ollama},
            ),
            FakeModelStore(
                ModelStoreKind.HUGGING_FACE,
                local_artifacts={
                    configuration.speech_to_text_model: _hugging_face_artifact(
                        tmp_path,
                        fill,
                    )
                },
            ),
        ).resolve_models(configuration)

    # Act
    first = resolve_with_speech("2")
    second = resolve_with_speech("3")

    # Assert
    assert first.for_role(ModelRole.CANDIDATE_ANNOTATION).semantic_input() == (
        second.for_role(ModelRole.CANDIDATE_ANNOTATION).semantic_input()
    )
    assert first.for_role(ModelRole.SPEECH_TO_TEXT).semantic_input() != (
        second.for_role(ModelRole.SPEECH_TO_TEXT).semantic_input()
    )


def test_provenance_does_not_expose_model_path_or_credentials(tmp_path: Path) -> None:
    """model provenanceにartifact絶対pathやcredentialが公開されないこと。

    Arrange:
        - credential風文字列を含む絶対path上のlocal STT artifactが用意される
    Act:
        - model解決結果がJSON provenanceへ変換される
    Assert:
        - pathとcredentialが含まれず4つのidentity項目が分離されること
    """
    # Arrange
    configuration = _configuration(tmp_path, auto_upgrade=False)
    ollama = _ollama_artifact("1")
    secret_path = tmp_path / "token-secret" / "snapshots" / ("2" * 40)
    hugging_face = ModelArtifact(
        identity=ResolvedModelIdentity(
            ModelStoreKind.HUGGING_FACE,
            "2" * 40,
        ),
        canonical_name=configuration.speech_to_text_model,
        runtime_identity=ModelRuntimeIdentity(
            ModelStoreKind.HUGGING_FACE,
            "0.36.2",
        ),
        location=secret_path,
    )
    runtime = _runtime(
        FakeModelStore(
            ModelStoreKind.OLLAMA,
            local_artifacts={configuration.scene_catalog_model: ollama},
        ),
        FakeModelStore(
            ModelStoreKind.HUGGING_FACE,
            local_artifacts={configuration.speech_to_text_model: hugging_face},
        ),
    )

    # Act
    provenance = runtime.resolve_models(configuration).provenance()
    serialized = json.dumps(provenance)

    # Assert
    assert str(tmp_path) not in serialized
    assert "token-secret" not in serialized
    speech = provenance[ModelRole.SPEECH_TO_TEXT.value]
    assert set(speech) == {
        "canonical_name",
        "configured_name",
        "execution_identity",
        "local_identity_before_update",
        "runtime_identity",
        "store",
        "update_status",
    }


def _configuration(
    tmp_path: Path,
    *,
    auto_upgrade: bool,
) -> EffectiveConfiguration:
    return EffectiveConfiguration(
        video_input_folder=tmp_path / "videos",
        output_folder=tmp_path / "output",
        models_auto_upgrade=auto_upgrade,
    )


def _runtime(
    ollama_store: FakeModelStore,
    hugging_face_store: FakeModelStore,
) -> ModelLifecycleRuntime:
    return ModelLifecycleRuntime(
        ollama_store_factory=lambda _configuration: ollama_store,
        hugging_face_store_factory=lambda _configuration: hugging_face_store,
    )


def _ollama_artifact(fill: str) -> ModelArtifact:
    return ModelArtifact(
        identity=ResolvedModelIdentity(
            ModelStoreKind.OLLAMA,
            "sha256:" + fill * 64,
        ),
        canonical_name="qwen3-vl:8b-instruct",
        runtime_identity=ModelRuntimeIdentity(ModelStoreKind.OLLAMA, "0.31.2"),
        location=None,
    )


def _hugging_face_artifact(tmp_path: Path, fill: str) -> ModelArtifact:
    return ModelArtifact(
        identity=ResolvedModelIdentity(
            ModelStoreKind.HUGGING_FACE,
            fill * 40,
        ),
        canonical_name="dropbox-dash/faster-whisper-large-v3-turbo",
        runtime_identity=ModelRuntimeIdentity(
            ModelStoreKind.HUGGING_FACE,
            "0.36.2",
        ),
        location=tmp_path / "model-store" / "snapshots" / (fill * 40),
    )
