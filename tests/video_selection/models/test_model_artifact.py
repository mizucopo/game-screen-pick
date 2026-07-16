from pathlib import Path

import pytest

from src.video_selection.models.model_artifact import ModelArtifact
from src.video_selection.models.model_runtime_identity import ModelRuntimeIdentity
from src.video_selection.models.model_store_kind import ModelStoreKind
from src.video_selection.models.resolved_model_identity import ResolvedModelIdentity


def test_store_specific_artifact_shape_is_accepted(tmp_path: Path) -> None:
    """store固有のidentity、runtime、locationの組が受理されること。

    Arrange:
        - OllamaとHugging Faceの完全artifact metadataが用意される
    Act:
        - store別Model Artifactが構築される
    Assert:
        - canonical名と非公開locationが保持されること
    """
    # Arrange
    snapshot = tmp_path / ("b" * 40)

    # Act
    ollama = ModelArtifact(
        identity=ResolvedModelIdentity(
            ModelStoreKind.OLLAMA,
            "sha256:" + "a" * 64,
        ),
        canonical_name="qwen3-vl:latest",
        runtime_identity=ModelRuntimeIdentity(ModelStoreKind.OLLAMA, "0.31.2"),
        location=None,
    )
    hugging_face = ModelArtifact(
        identity=ResolvedModelIdentity(ModelStoreKind.HUGGING_FACE, "b" * 40),
        canonical_name="org/model",
        runtime_identity=ModelRuntimeIdentity(
            ModelStoreKind.HUGGING_FACE,
            "0.36.2",
        ),
        location=snapshot,
    )

    # Assert
    assert ollama.canonical_name == "qwen3-vl:latest"
    assert hugging_face.location == snapshot


def test_artifact_runtime_store_mismatch_is_rejected() -> None:
    """artifactと異なるstoreのruntime identityが拒否されること。

    Arrange:
        - Ollama digestとHugging Face runtime identityが用意される
    Act:
        - 不一致なModel Artifactの構築が試行される
    Assert:
        - store contractのvalidation errorになること
    """
    # Arrange
    identity = ResolvedModelIdentity(
        ModelStoreKind.OLLAMA,
        "sha256:" + "a" * 64,
    )
    runtime_identity = ModelRuntimeIdentity(ModelStoreKind.HUGGING_FACE, "0.36.2")

    # Act
    # Assert
    with pytest.raises(ValueError, match="store kind"):
        ModelArtifact(
            identity=identity,
            canonical_name="qwen3-vl:latest",
            runtime_identity=runtime_identity,
            location=None,
        )
