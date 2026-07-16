from src.video_selection.models.model_role import ModelRole
from src.video_selection.models.model_runtime_error import ModelRuntimeError
from src.video_selection.models.model_runtime_failure_reason import (
    ModelRuntimeFailureReason,
)


def test_runtime_error_exposes_only_stable_reason_and_role() -> None:
    """Model Runtime errorが安全な説明とstable fieldだけを公開すること。

    Arrange:
        - model store unavailable reasonと対象roleが用意される
    Act:
        - Model Runtime errorが構築される
    Assert:
        - reasonとroleを保持し外部detailを含まないmessageになること
    """
    # Arrange
    reason = ModelRuntimeFailureReason.MODEL_STORE_UNAVAILABLE
    role = ModelRole.SPEECH_TO_TEXT

    # Act
    error = ModelRuntimeError(reason, role)

    # Assert
    assert error.reason is reason
    assert error.role is role
    assert str(error) == "speech_to_text: model storeの状態を確認できませんでした"
