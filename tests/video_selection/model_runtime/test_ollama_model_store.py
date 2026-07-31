import traceback
from collections.abc import Mapping

import pytest

from src.video_selection.model_runtime.ollama_model_store import OllamaModelStore
from src.video_selection.models.model_artifact_invalid_error import (
    ModelArtifactInvalidError,
)
from src.video_selection.models.model_capability import ModelCapability
from src.video_selection.models.model_requirement import ModelRequirement
from src.video_selection.models.model_role import ModelRole
from src.video_selection.models.model_store_kind import ModelStoreKind
from src.video_selection.models.model_store_unavailable_error import (
    ModelStoreUnavailableError,
)


def test_local_pull_and_capability_use_documented_ollama_api() -> None:
    """local解決、pull、capability検証がdocumented APIで行われること。

    Arrange:
        - pull前後で異なる完全digestを返すfake Ollama APIが用意される
        - visionと十分なcontext lengthを持つshow responseが用意される
    Act:
        - local解決、同期、capability検証が順に実行される
    Assert:
        - post-pull digestとserver versionがfreezeされること
        - pull payloadに手入力hashが含まれないこと
    """
    # Arrange
    requests: list[tuple[str, str, Mapping[str, object] | None, float]] = []
    pulled = False

    def requester(
        method: str,
        url: str,
        payload: Mapping[str, object] | None,
        timeout: float,
    ) -> object:
        nonlocal pulled
        requests.append((method, url, payload, timeout))
        if url.endswith("/api/version"):
            return {"version": "0.31.2"}
        if url.endswith("/api/tags"):
            fill = "b" if pulled else "a"
            return {
                "models": [
                    {
                        "name": "qwen3-vl:8b-instruct",
                        "model": "qwen3-vl:8b-instruct",
                        "digest": fill * 64,
                    }
                ]
            }
        if url.endswith("/api/pull"):
            pulled = True
            return {"status": "success"}
        if url.endswith("/api/show"):
            return {
                "capabilities": ["completion", "vision"],
                "model_info": {"qwen3vl.context_length": 131072},
            }
        if url.endswith("/api/chat"):
            return {"message": {"content": '{"ready": true}'}}
        raise AssertionError(url)

    store = OllamaModelStore(
        "http://ollama.example:11434/",
        timeout_seconds=45.0,
        requester=requester,
    )
    requirement = _requirement()

    # Act
    before = store.resolve_local(requirement)
    after = store.synchronize(requirement)
    store.validate(after, requirement)
    store.confirm_current_identity(after, requirement)

    # Assert
    assert before is not None
    assert before.identity.identifier == "ollama:sha256:" + "a" * 64
    assert after.identity.identifier == "ollama:sha256:" + "b" * 64
    assert after.runtime_identity.identifier == "ollama:0.31.2"
    pull_requests = [item for item in requests if item[1].endswith("/api/pull")]
    assert pull_requests == [
        (
            "POST",
            "http://ollama.example:11434/api/pull",
            {"model": "qwen3-vl:8b-instruct", "stream": False},
            45.0,
        )
    ]
    pull_payload = pull_requests[0][2]
    assert pull_payload is not None
    assert "digest" not in pull_payload
    assert "revision" not in pull_payload
    capability_requests = [item for item in requests if item[1].endswith("/api/chat")]
    assert len(capability_requests) == 1
    capability_payload = capability_requests[0][2]
    assert capability_payload is not None
    assert capability_payload["model"] == "qwen3-vl:8b-instruct"
    assert capability_payload["keep_alive"] == 0
    assert capability_payload["options"] == {
        "temperature": 0,
        "num_ctx": 32768,
    }


@pytest.mark.parametrize(
    "show_response",
    [
        {
            "capabilities": ["completion"],
            "model_info": {"qwen3vl.context_length": 131072},
        },
        {
            "capabilities": ["completion", "vision"],
            "model_info": {"qwen3vl.context_length": 8192},
        },
    ],
)
def test_missing_vision_or_context_capability_is_rejected(
    show_response: dict[str, object],
) -> None:
    """visionまたは要求context capabilityを欠くmodelが拒否されること。

    Arrange:
        - 完全digestと不足するcapability responseが用意される
    Act:
        - Ollama artifactのcapability検証が実行される
    Assert:
        - artifact invalid errorになること
    """

    # Arrange
    def requester(
        _method: str,
        url: str,
        _payload: Mapping[str, object] | None,
        _timeout: float,
    ) -> object:
        if url.endswith("/api/version"):
            return {"version": "0.31.2"}
        if url.endswith("/api/tags"):
            return {
                "models": [
                    {
                        "name": "qwen3-vl:8b-instruct",
                        "digest": "sha256:" + "a" * 64,
                    }
                ]
            }
        return show_response

    store = OllamaModelStore(
        "http://localhost:11434",
        timeout_seconds=60.0,
        requester=requester,
    )
    requirement = _requirement()
    artifact = store.resolve_local(requirement)
    assert artifact is not None

    # Act
    # Assert
    with pytest.raises(ModelArtifactInvalidError, match="capability"):
        store.validate(artifact, requirement)


def test_malformed_digest_and_transport_detail_are_not_exposed() -> None:
    """不正digestとtransport detailが安全なadapter errorへ変換されること。

    Arrange:
        - 不完全digest responseと秘密を含むtransport failureが用意される
    Act:
        - 各storeでlocal model解決が試行される
    Assert:
        - 不完全identityが拒否され秘密がpublic messageへ出ないこと
    """

    # Arrange
    def malformed_requester(
        _method: str,
        url: str,
        _payload: Mapping[str, object] | None,
        _timeout: float,
    ) -> object:
        if url.endswith("/api/version"):
            return {"version": "0.31.2"}
        return {
            "models": [
                {
                    "name": "qwen3-vl:8b-instruct",
                    "digest": "sha256:partial",
                }
            ]
        }

    def failing_requester(
        _method: str,
        _url: str,
        _payload: Mapping[str, object] | None,
        _timeout: float,
    ) -> object:
        raise RuntimeError("token-secret /private/model-store")

    malformed = OllamaModelStore(
        "http://localhost:11434",
        timeout_seconds=60.0,
        requester=malformed_requester,
    )
    unavailable = OllamaModelStore(
        "http://localhost:11434",
        timeout_seconds=60.0,
        requester=failing_requester,
    )

    # Act
    # Assert
    with pytest.raises(ModelArtifactInvalidError):
        malformed.resolve_local(_requirement())
    with pytest.raises(ModelStoreUnavailableError) as captured:
        unavailable.resolve_local(_requirement())
    assert "token-secret" not in str(captured.value)
    assert "/private/model-store" not in str(captured.value)
    formatted = "".join(
        traceback.format_exception(captured.type, captured.value, captured.tb)
    )
    assert "token-secret" not in formatted
    assert "/private/model-store" not in formatted


def test_structured_output_capability_failure_is_rejected() -> None:
    """strict schema responseを生成できないOllama modelが拒否されること。

    Arrange:
        - visionとcontextは満たすがschema外responseを返すAPIが用意される
    Act:
        - Ollama artifactのcapability検証が実行される
    Assert:
        - structured output capability failureになること
    """

    # Arrange
    def requester(
        _method: str,
        url: str,
        _payload: Mapping[str, object] | None,
        _timeout: float,
    ) -> object:
        if url.endswith("/api/version"):
            return {"version": "0.31.2"}
        if url.endswith("/api/tags"):
            return {
                "models": [
                    {
                        "name": "qwen3-vl:8b-instruct",
                        "digest": "sha256:" + "a" * 64,
                    }
                ]
            }
        if url.endswith("/api/show"):
            return {
                "capabilities": ["completion", "vision"],
                "model_info": {"qwen3vl.context_length": 131072},
            }
        return {"message": {"content": '{"ready": false}'}}

    store = OllamaModelStore(
        "http://localhost:11434",
        timeout_seconds=60.0,
        requester=requester,
    )
    artifact = store.resolve_local(_requirement())
    assert artifact is not None

    # Act
    # Assert
    with pytest.raises(ModelArtifactInvalidError, match="structured output"):
        store.validate(artifact, _requirement())


def test_changed_tag_digest_is_rejected_after_capability_validation() -> None:
    """capability検証中にmutable tagが移動したartifactがfreezeされないこと。

    Arrange:
        - local解決時と最終確認時で異なるdigestを返すAPIが用意される
        - 両digestのmodelがcapability probeへ応答するようにされる
    Act:
        - 解決済みartifactのcapabilityが検証される
    Assert:
        - 最終identity確認でtag移動がartifact invalidとして拒否されること
    """
    # Arrange
    tag_resolutions = 0

    def requester(
        _method: str,
        url: str,
        _payload: Mapping[str, object] | None,
        _timeout: float,
    ) -> object:
        nonlocal tag_resolutions
        if url.endswith("/api/version"):
            return {"version": "0.31.2"}
        if url.endswith("/api/tags"):
            tag_resolutions += 1
            fill = "a" if tag_resolutions == 1 else "b"
            return {
                "models": [
                    {
                        "name": "qwen3-vl:8b-instruct",
                        "digest": "sha256:" + fill * 64,
                    }
                ]
            }
        if url.endswith("/api/show"):
            return {
                "capabilities": ["completion", "vision"],
                "model_info": {"qwen3vl.context_length": 131072},
            }
        return {"message": {"content": '{"ready": true}'}}

    store = OllamaModelStore(
        "http://localhost:11434",
        timeout_seconds=60.0,
        requester=requester,
    )
    requirement = _requirement()
    artifact = store.resolve_local(requirement)
    assert artifact is not None

    # Act
    store.validate(artifact, requirement)

    # Assert
    with pytest.raises(ModelArtifactInvalidError, match="変更"):
        store.confirm_current_identity(artifact, requirement)


def test_unsupported_server_is_rejected_before_pull_mutation() -> None:
    """最低version未満のOllama serverでpullが開始されないこと。

    Arrange:
        - project floor未満のserver versionを返すAPIが用意される
    Act:
        - configured modelの同期が試行される
    Assert:
        - version errorとなりpull requestが送信されないこと
    """
    # Arrange
    requested_paths: list[str] = []

    def requester(
        _method: str,
        url: str,
        _payload: Mapping[str, object] | None,
        _timeout: float,
    ) -> object:
        requested_paths.append(url)
        if url.endswith("/api/version"):
            return {"version": "0.30.0"}
        return {"status": "success"}

    store = OllamaModelStore(
        "http://localhost:11434",
        timeout_seconds=60.0,
        requester=requester,
    )

    # Act
    # Assert
    with pytest.raises(ModelArtifactInvalidError, match="0.31.2"):
        store.synchronize(_requirement())
    assert not any(path.endswith("/api/pull") for path in requested_paths)


def test_unsafe_server_version_is_rejected_as_artifact_invalid() -> None:
    """unsafeなOllama version responseがadapter errorへ変換されること。

    Arrange:
        - path風suffixを含む最低version以上のserver responseが用意される
    Act:
        - configured modelのlocal解決が試行される
    Assert:
        - raw versionを公開しないartifact invalid errorになること
    """
    # Arrange
    unsafe_version = "0.31.2/token-secret"
    store = OllamaModelStore(
        "http://localhost:11434",
        timeout_seconds=60.0,
        requester=lambda _method, _url, _payload, _timeout: {"version": unsafe_version},
    )

    # Act
    # Assert
    with pytest.raises(ModelArtifactInvalidError) as captured:
        store.resolve_local(_requirement())
    assert unsafe_version not in str(captured.value)


def _requirement() -> ModelRequirement:
    return ModelRequirement(
        role=ModelRole.SCENE_CATALOG,
        store_kind=ModelStoreKind.OLLAMA,
        configured_name="qwen3-vl:8b-instruct",
        capability=ModelCapability.VISION_STRUCTURED_OUTPUT,
        minimum_context_length=32768,
    )
