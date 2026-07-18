import pytest

from src.video_selection.models.vision_inference_diagnostics import (
    VisionInferenceDiagnostics,
)


def test_canonical_ollama_diagnostics_is_accepted() -> None:
    """canonicalなOllama identityを持つ診断が受理されること。

    Arrange:
        - 完全model digestと安全なruntime versionが用意される
    Act:
        - Vision推論診断が構築される
    Assert:
        - identityが変更されず保持されること
    """
    # Arrange
    model_identity = "ollama:sha256:" + "a" * 64
    runtime_identity = "ollama:0.31.2"

    # Act
    diagnostics = _diagnostics(model_identity, runtime_identity)

    # Assert
    assert diagnostics.model_identity == model_identity
    assert diagnostics.runtime_identity == runtime_identity


def test_canonical_hugging_face_diagnostics_is_accepted() -> None:
    """canonicalなHugging Face identityを持つ診断が受理されること。

    Arrange:
        - 完全commit SHAと安全なruntime versionが用意される
    Act:
        - Vision推論診断が構築される
    Assert:
        - identityが変更されず保持されること
    """
    # Arrange
    model_identity = "hf:" + "b" * 40
    runtime_identity = "huggingface-hub:0.30.0"

    # Act
    diagnostics = _diagnostics(model_identity, runtime_identity)

    # Assert
    assert diagnostics.model_identity == model_identity
    assert diagnostics.runtime_identity == runtime_identity


@pytest.mark.parametrize(
    ("model_identity", "runtime_identity"),
    (
        ("/private/model", "ollama:0.31.2"),
        ("ollama:sha256:" + "a" * 64, "/private/ollama"),
        ("hf:" + "b" * 40, "ollama:0.31.2"),
    ),
)
def test_noncanonical_or_mismatched_diagnostic_identity_is_rejected(
    model_identity: str,
    runtime_identity: str,
) -> None:
    """非canonicalまたはstore不一致のdiagnostic identityが拒否されること。

    Arrange:
        - pathまたは異なるstoreを含むmodel/runtime identityが用意される
    Act:
        - Vision推論診断が構築される
    Assert:
        - privacy-safeなcanonical identityではないため拒否されること
    """
    # Arrange

    # Act
    # Assert
    with pytest.raises(ValueError, match="Vision inference diagnostics"):
        _diagnostics(model_identity, runtime_identity)


@pytest.mark.parametrize("attempt_count", (3, 4))
def test_aggregate_candidate_attempt_count_is_accepted(attempt_count: int) -> None:
    """注釈と専用確認を集約した推論回数が受理されること。

    Arrange:
        - 注釈と専用確認の合計を表す3回または4回が用意される
    Act:
        - Vision推論診断が構築される
    Assert:
        - 集約された推論回数が変更されず保持されること
    """
    # Arrange

    # Act
    diagnostics = _diagnostics(
        "ollama:sha256:" + "a" * 64,
        "ollama:0.31.2",
        attempt_count=attempt_count,
    )

    # Assert
    assert diagnostics.attempt_count == attempt_count


def test_attempt_count_above_aggregate_limit_is_rejected() -> None:
    """集約上限を超える推論回数が拒否されること。

    Arrange:
        - 注釈と専用確認の合計上限を超える5回が用意される
    Act:
        - Vision推論診断が構築される
    Assert:
        - 不正な診断として拒否されること
    """
    # Arrange

    # Act
    # Assert
    with pytest.raises(ValueError, match="Vision inference diagnostics"):
        _diagnostics(
            "ollama:sha256:" + "a" * 64,
            "ollama:0.31.2",
            attempt_count=5,
        )


def _diagnostics(
    model_identity: str,
    runtime_identity: str,
    *,
    attempt_count: int = 1,
) -> VisionInferenceDiagnostics:
    return VisionInferenceDiagnostics(
        request_fingerprint="c" * 64,
        model_name="qwen3-vl:8b-instruct",
        model_identity=model_identity,
        runtime_identity=runtime_identity,
        prompt_version="prompt-v1",
        schema_version="schema-v1",
        stage_contract_version="stage-v1",
        retry_policy_version="retry-v1",
        cache_hit=False,
        attempt_count=attempt_count,
        validation_code=None,
        image_count=1,
        context_cue_count=0,
        duration_seconds=0.1,
        prompt_eval_count=10,
        eval_count=5,
        done_reason="stop",
    )
