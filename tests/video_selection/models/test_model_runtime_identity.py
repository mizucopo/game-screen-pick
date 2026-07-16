import pytest

from src.video_selection.models.model_runtime_identity import ModelRuntimeIdentity
from src.video_selection.models.model_store_kind import ModelStoreKind


@pytest.mark.parametrize(
    ("store_kind", "version", "expected"),
    (
        (ModelStoreKind.OLLAMA, "0.31.2", "ollama:0.31.2"),
        (
            ModelStoreKind.HUGGING_FACE,
            "0.36.2",
            "huggingface-hub:0.36.2",
        ),
    ),
)
def test_store_runtime_version_is_canonicalized(
    store_kind: ModelStoreKind,
    version: str,
    expected: str,
) -> None:
    """storeとversionがcanonical runtime identityへ変換されること。

    Arrange:
        - model storeと安全なruntime versionが用意される
    Act:
        - Model Runtime Identityが構築される
    Assert:
        - store間で衝突しないidentifierが返されること
    """
    # Arrange
    runtime_version = version

    # Act
    identity = ModelRuntimeIdentity(store_kind, runtime_version)

    # Assert
    assert identity.identifier == expected


@pytest.mark.parametrize("version", ("", " token", "1/../../secret", "v:1"))
def test_unsafe_runtime_version_is_rejected(version: str) -> None:
    """空またはpath・credential化し得るruntime versionが拒否されること。

    Arrange:
        - runtime identity contractを満たさないversionが用意される
    Act:
        - Model Runtime Identityの構築が試行される
    Assert:
        - raw versionを表示しないvalidation errorになること
    """
    # Arrange
    unsafe_version = version

    # Act
    # Assert
    with pytest.raises(ValueError, match="runtime version") as captured:
        ModelRuntimeIdentity(ModelStoreKind.OLLAMA, unsafe_version)
    if unsafe_version:
        assert unsafe_version not in str(captured.value)
