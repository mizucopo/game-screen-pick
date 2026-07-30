from pathlib import Path

import pytest

from src.video_selection.models.model_role import ModelRole
from src.video_selection.models.model_runtime_identity import ModelRuntimeIdentity
from src.video_selection.models.model_store_kind import ModelStoreKind
from src.video_selection.models.model_update_status import ModelUpdateStatus
from src.video_selection.models.resolved_model import ResolvedModel
from src.video_selection.models.resolved_model_identity import ResolvedModelIdentity


def test_model_resolution_separates_semantics_from_diagnostics(tmp_path: Path) -> None:
    """実行semanticsと更新diagnostic・private locationが分離されること。

    Arrange:
        - 更新済みHugging Face modelのrun別fieldが用意される
    Act:
        - semantic inputとprovenanceが取得される
    Assert:
        - semantic inputが実行identityだけを持つこと
        - provenanceに更新情報を持ちabsolute pathを含まないこと
    """
    # Arrange
    before = ResolvedModelIdentity(ModelStoreKind.HUGGING_FACE, "a" * 40)
    execution = ResolvedModelIdentity(ModelStoreKind.HUGGING_FACE, "b" * 40)
    model = ResolvedModel(
        role=ModelRole.SPEECH_TO_TEXT,
        configured_name="alias/model",
        canonical_name="canonical/model",
        local_identity_before_update=before,
        update_status=ModelUpdateStatus.UPDATED,
        execution_identity=execution,
        runtime_identity=ModelRuntimeIdentity(
            ModelStoreKind.HUGGING_FACE,
            "0.36.2",
        ),
        artifact_location=tmp_path / "token-secret" / ("b" * 40),
    )

    # Act
    semantic_input = model.semantic_input()
    provenance = model.provenance()

    # Assert
    assert semantic_input == {
        "execution_identity": "hf:" + "b" * 40,
        "runtime_identity": "huggingface-hub:0.36.2",
        "store": "hugging_face",
    }
    assert provenance["local_identity_before_update"] == "hf:" + "a" * 40
    assert provenance["update_status"] == "updated"
    assert str(tmp_path) not in str(provenance)
    assert "token-secret" not in str(provenance)


def test_model_role_store_mismatch_is_rejected() -> None:
    """model roleと異なるexecution storeが拒否されること。

    Arrange:
        - STT roleとOllama execution identityが用意される
    Act:
        - 不一致なResolved Modelの構築が試行される
    Assert:
        - role/store contractのvalidation errorになること
    """
    # Arrange
    identity = ResolvedModelIdentity(
        ModelStoreKind.OLLAMA,
        "sha256:" + "a" * 64,
    )

    # Act
    # Assert
    with pytest.raises(ValueError, match="roleとstore kind"):
        ResolvedModel(
            role=ModelRole.SPEECH_TO_TEXT,
            configured_name="qwen3-vl:latest",
            canonical_name="qwen3-vl:latest",
            local_identity_before_update=identity,
            update_status=ModelUpdateStatus.NOT_REQUESTED,
            execution_identity=identity,
            runtime_identity=ModelRuntimeIdentity(ModelStoreKind.OLLAMA, "0.31.2"),
            artifact_location=None,
        )


def test_configured_alias_does_not_change_semantic_model_identity() -> None:
    """同じartifact/runtimeを指すalias変更でStage依存が変わらないこと。

    Arrange:
        - configured nameだけが異なる同一Ollama identityのmodelが用意される
    Act:
        - 両modelのsemantic inputが取得される
    Assert:
        - semantic inputは一致しaliasはprovenanceだけに保持されること
    """
    # Arrange
    identity = ResolvedModelIdentity(
        ModelStoreKind.OLLAMA,
        "sha256:" + "a" * 64,
    )
    runtime = ModelRuntimeIdentity(ModelStoreKind.OLLAMA, "0.31.2")

    def model(alias: str) -> ResolvedModel:
        return ResolvedModel(
            role=ModelRole.SCENE_CATALOG,
            configured_name=alias,
            canonical_name="registry/model:latest",
            local_identity_before_update=identity,
            update_status=ModelUpdateStatus.NOT_REQUESTED,
            execution_identity=identity,
            runtime_identity=runtime,
            artifact_location=None,
        )

    # Act
    first = model("registry/model").semantic_input()
    second_model = model("registry/model:latest")
    second = second_model.semantic_input()

    # Assert
    assert first == second
    assert "configured_name" not in first
    assert second_model.provenance()["configured_name"] == "registry/model:latest"
