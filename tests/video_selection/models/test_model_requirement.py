import pytest

from src.video_selection.models.model_capability import ModelCapability
from src.video_selection.models.model_requirement import ModelRequirement
from src.video_selection.models.model_role import ModelRole
from src.video_selection.models.model_store_kind import ModelStoreKind


@pytest.mark.parametrize(
    "role",
    (ModelRole.SCENE_CATALOG, ModelRole.CANDIDATE_ANNOTATION),
)
def test_vision_roles_require_ollama_structured_output(role: ModelRole) -> None:
    """vision roleがOllama structured output要求として構築されること。

    Arrange:
        - vision roleと必要context lengthが用意される
    Act:
        - Model Requirementが構築される
    Assert:
        - role固有の要求値が保持されること
    """
    # Arrange
    minimum_context_length = 32768

    # Act
    requirement = ModelRequirement(
        role=role,
        store_kind=ModelStoreKind.OLLAMA,
        configured_name="qwen3-vl:latest",
        capability=ModelCapability.VISION_STRUCTURED_OUTPUT,
        minimum_context_length=minimum_context_length,
    )

    # Assert
    assert requirement.minimum_context_length == minimum_context_length


def test_speech_role_requires_hugging_face_execution_profile() -> None:
    """speech roleがHugging Face実行profile付きで構築されること。

    Arrange:
        - STT model名、device、compute typeが用意される
    Act:
        - Model Requirementが構築される
    Assert:
        - speech capabilityと実行profileが保持されること
    """
    # Arrange
    configured_name = "org/model"

    # Act
    requirement = ModelRequirement(
        role=ModelRole.SPEECH_TO_TEXT,
        store_kind=ModelStoreKind.HUGGING_FACE,
        configured_name=configured_name,
        capability=ModelCapability.SPEECH_TO_TEXT,
        device="cuda",
        compute_type="float16",
    )

    # Assert
    assert requirement.configured_name == configured_name


def test_vision_capability_cannot_be_assigned_to_speech_role() -> None:
    """speech roleへvision capabilityが割り当てられないこと。

    Arrange:
        - speech roleとOllama vision要求が用意される
    Act:
        - 不一致なModel Requirementの構築が試行される
    Assert:
        - vision requirementのvalidation errorになること
    """
    # Arrange
    role = ModelRole.SPEECH_TO_TEXT

    # Act
    # Assert
    with pytest.raises(ValueError, match="vision model requirement"):
        ModelRequirement(
            role=role,
            store_kind=ModelStoreKind.OLLAMA,
            configured_name="qwen3-vl:latest",
            capability=ModelCapability.VISION_STRUCTURED_OUTPUT,
            minimum_context_length=32768,
        )
