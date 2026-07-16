import pytest

from src.video_selection.models.model_store_kind import ModelStoreKind
from src.video_selection.models.resolved_model_identity import ResolvedModelIdentity


def test_complete_store_identity_is_canonicalized() -> None:
    """完全digestとcommit SHAがstore別canonical identityへ変換されること。

    Arrange:
        - 完全なOllama digestとHugging Face commit SHAが用意される
    Act:
        - store別のResolved Model Identityが構築される
    Assert:
        - 衝突しないcanonical identifierが返されること
    """
    # Arrange
    ollama_digest = "sha256:" + "a" * 64
    hugging_face_commit = "b" * 40

    # Act
    ollama = ResolvedModelIdentity(ModelStoreKind.OLLAMA, ollama_digest)
    hugging_face = ResolvedModelIdentity(
        ModelStoreKind.HUGGING_FACE,
        hugging_face_commit,
    )

    # Assert
    assert ollama.identifier == f"ollama:{ollama_digest}"
    assert hugging_face.identifier == f"hf:{hugging_face_commit}"


@pytest.mark.parametrize(
    ("store_kind", "value"),
    [
        (ModelStoreKind.OLLAMA, "sha256:short"),
        (ModelStoreKind.OLLAMA, "a" * 64),
        (ModelStoreKind.HUGGING_FACE, "short"),
        (ModelStoreKind.HUGGING_FACE, "G" * 40),
    ],
)
def test_incomplete_or_malformed_identity_is_rejected(
    store_kind: ModelStoreKind,
    value: str,
) -> None:
    """不完全または不正なmodel identityが拒否されること。

    Arrange:
        - store contractを満たさないidentity値が用意される
    Act:
        - Resolved Model Identityの構築が試行される
    Assert:
        - raw identityを表示せずvalidation errorになること
    """
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="完全なmodel identity") as captured:
        ResolvedModelIdentity(store_kind, value)
    assert value not in str(captured.value)
