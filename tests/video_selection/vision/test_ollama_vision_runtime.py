import json
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from email.message import Message
from email.utils import format_datetime
from fractions import Fraction
from threading import Event, Thread
from typing import cast
from urllib.error import HTTPError

import pytest

from src.video_selection.models.candidate_annotation import (
    candidate_annotation_free_text_is_safe,
)
from src.video_selection.models.candidate_annotation_request import (
    CandidateAnnotationRequest,
)
from src.video_selection.models.candidate_moment import CandidateMoment
from src.video_selection.models.context_cue import ContextCue
from src.video_selection.models.frame_candidate import FrameCandidate
from src.video_selection.models.model_artifact import ModelArtifact
from src.video_selection.models.model_role import ModelRole
from src.video_selection.models.model_runtime_identity import ModelRuntimeIdentity
from src.video_selection.models.model_store_kind import ModelStoreKind
from src.video_selection.models.model_update_status import ModelUpdateStatus
from src.video_selection.models.report_value import string_looks_private
from src.video_selection.models.resolved_model import ResolvedModel
from src.video_selection.models.resolved_model_identity import ResolvedModelIdentity
from src.video_selection.models.scene_catalog import SceneCatalog
from src.video_selection.models.scene_catalog_entry import SceneCatalogEntry
from src.video_selection.models.scene_catalog_request import SceneCatalogRequest
from src.video_selection.models.vision_runtime_error import VisionRuntimeError
from src.video_selection.models.vision_runtime_failure_reason import (
    VisionRuntimeFailureReason,
)
from src.video_selection.services.gpu_work_coordinator import GpuWorkCoordinator
from src.video_selection.vision.ollama_vision_runtime import OllamaVisionRuntime


def test_inference_waits_for_shared_gpu_lease() -> None:
    """Ollama推論が共有GPU leaseを取得してからrequestされること。

    Arrange:
        - STT相当workが保持中の実coordinatorとVision Runtimeが用意される
    Act:
        - 別threadからScene Catalog推論が要求され、先行leaseが解放される
    Assert:
        - 解放前はOllamaへrequestされず、解放後に推論が完了すること
    """
    # Arrange
    coordinator = GpuWorkCoordinator()
    lease_started = Event()
    release_lease = Event()
    request_count = 0
    failures: list[BaseException] = []

    def requester(
        _method: str,
        _url: str,
        _payload: Mapping[str, object] | None,
        _timeout: float,
    ) -> object:
        nonlocal request_count
        request_count += 1
        return _response(_catalog_payload())

    runtime = OllamaVisionRuntime(
        "http://localhost:11434",
        timeout_seconds=60.0,
        requester=requester,
        sleeper=lambda _seconds: None,
        model_state_resolver=_resolved_artifact,
        gpu_coordinator=coordinator,
    )
    request = SceneCatalogRequest(
        representatives=(FrameCandidate("frame-a", b"image-a"),),
        selection_intent="ブログ画像を分類する",
    )

    def hold_gpu_lease() -> None:
        def wait_for_release() -> None:
            lease_started.set()
            if not release_lease.wait(timeout=1.0):
                msg = "GPU leaseを解放できませんでした"
                raise RuntimeError(msg)

        try:
            coordinator.run("speech_to_text", wait_for_release)
        except BaseException as error:
            failures.append(error)

    def infer() -> None:
        try:
            runtime.create_scene_catalog(
                request,
                _resolved_model(ModelRole.SCENE_CATALOG),
                num_ctx=32768,
            )
        except BaseException as error:
            failures.append(error)

    holder = Thread(target=hold_gpu_lease)
    worker = Thread(target=infer)

    # Act
    holder.start()
    assert lease_started.wait(timeout=1.0)
    worker.start()
    worker.join(timeout=0.05)
    blocked_before_release = worker.is_alive() and request_count == 0
    release_lease.set()
    holder.join(timeout=1.0)
    worker.join(timeout=1.0)

    # Assert
    assert blocked_before_release is True
    assert failures == []
    assert request_count == 1
    assert holder.is_alive() is False
    assert worker.is_alive() is False


def test_scene_catalog_uses_strict_documented_ollama_request() -> None:
    """Scene Catalogが画像付きstrict schema requestから生成されること。

    Arrange:
        - 3件のsceneを返すfake Ollama APIと2枚の代表画像が用意される
    Act:
        - Scene Catalog推論が実行される
    Assert:
        - JSON Schema object、temperature 0、stream/think falseが送られること
        - token/runtime診断とdomain検証済みCatalogが返されること
    """
    # Arrange
    requests: list[Mapping[str, object]] = []

    def requester(
        _method: str,
        url: str,
        payload: Mapping[str, object] | None,
        _timeout: float,
    ) -> object:
        if url.endswith("/api/version"):
            return {"version": "0.31.2"}
        if url.endswith("/api/tags"):
            return {
                "models": [
                    {
                        "name": "qwen3-vl:8b-instruct",
                        "digest": "a" * 64,
                    }
                ]
            }
        assert payload is not None
        requests.append(payload)
        return _response(_catalog_payload())

    runtime = OllamaVisionRuntime(
        "http://ollama.example:11434/",
        timeout_seconds=45.0,
        requester=requester,
        sleeper=lambda _seconds: None,
    )
    request = SceneCatalogRequest(
        representatives=(
            FrameCandidate(identifier="frame-a", image_bytes=b"image-a"),
            FrameCandidate(identifier="frame-b", image_bytes=b"image-b"),
        ),
        selection_intent="ブログ本文を説明できる画像を分類する",
        scene_hint="RPGの探索と戦闘",
    )

    # Act
    catalog, diagnostics = runtime.create_scene_catalog(
        request,
        _resolved_model(ModelRole.SCENE_CATALOG),
        num_ctx=32768,
    )

    # Assert
    assert catalog.slugs == ("exploration", "battle", "other")
    assert diagnostics.attempt_count == 1
    assert diagnostics.prompt_eval_count == 123
    assert diagnostics.eval_count == 45
    assert diagnostics.done_reason == "stop"
    assert len(diagnostics.request_fingerprint) == 64
    assert len(requests) == 1
    payload = requests[0]
    assert payload["model"] == "qwen3-vl:8b-instruct"
    assert payload["stream"] is False
    assert payload["think"] is False
    assert payload["options"] == {"temperature": 0, "num_ctx": 32768}
    schema = payload["format"]
    assert isinstance(schema, dict)
    assert schema["additionalProperties"] is False
    assert "quality_score" not in json.dumps(schema)
    assert "final_score" not in json.dumps(schema)
    assert "selected" not in json.dumps(schema)
    messages = payload["messages"]
    assert isinstance(messages, list)
    assert messages[0]["images"] == ["aW1hZ2UtYQ==", "aW1hZ2UtYg=="]
    prompt = messages[0]["content"]
    assert isinstance(prompt, str)
    assert "recurring_gameplay=" in prompt
    assert "同じ画面構造を一時的な敵やエフェクトだけで別sceneへ分割しません" in prompt


def test_schema_failure_is_retried_once_with_stable_code() -> None:
    """schema invalidがraw responseを戻さず一度だけ修復再試行されること。

    Arrange:
        - 初回だけ未知fieldを含み2回目はvalidなresponseが用意される
    Act:
        - Scene Catalog推論が実行される
    Assert:
        - 同じ画像で2回だけrequestされstable codeだけが追加されること
    """
    # Arrange
    payloads: list[Mapping[str, object]] = []
    sleeps: list[float] = []

    def requester(
        _method: str,
        _url: str,
        payload: Mapping[str, object] | None,
        _timeout: float,
    ) -> object:
        assert payload is not None
        payloads.append(payload)
        response = _catalog_payload()
        if len(payloads) == 1:
            response["raw_reasoning"] = "secret chain of thought"
        return _response(response)

    runtime = OllamaVisionRuntime(
        "http://localhost:11434",
        timeout_seconds=60.0,
        requester=requester,
        sleeper=sleeps.append,
        model_state_resolver=_resolved_artifact,
    )
    request = SceneCatalogRequest(
        representatives=(FrameCandidate("frame-a", b"image-a"),),
        selection_intent="ブログ画像を分類する",
    )

    # Act
    catalog, diagnostics = runtime.create_scene_catalog(
        request,
        _resolved_model(ModelRole.SCENE_CATALOG),
        num_ctx=32768,
    )

    # Assert
    assert catalog.slugs[-1] == "other"
    assert diagnostics.attempt_count == 2
    assert diagnostics.validation_code == "scene_catalog_schema_invalid"
    assert sleeps == [1.0]
    assert (
        _first_message(payloads[0])["images"] == _first_message(payloads[1])["images"]
    )
    second_prompt = _first_message(payloads[1])["content"]
    assert isinstance(second_prompt, str)
    assert "scene_catalog_schema_invalid" in second_prompt
    assert "secret chain of thought" not in second_prompt


def test_truncated_response_is_retried_before_success() -> None:
    """token上限で打ち切られた応答が一度だけ再試行されること。

    Arrange:
        - 初回だけdone reasonがlengthとなるschema-valid responseが用意される
    Act:
        - Scene Catalog推論が実行される
    Assert:
        - 打ち切り応答が採用されずstable code付きで再試行されること
    """
    # Arrange
    payloads: list[Mapping[str, object]] = []

    def requester(
        _method: str,
        _url: str,
        payload: Mapping[str, object] | None,
        _timeout: float,
    ) -> object:
        assert payload is not None
        payloads.append(payload)
        response = _response(_catalog_payload())
        if len(payloads) == 1:
            response["done_reason"] = "length"
        return response

    runtime = OllamaVisionRuntime(
        "http://localhost:11434",
        timeout_seconds=60.0,
        requester=requester,
        sleeper=lambda _seconds: None,
        model_state_resolver=_resolved_artifact,
    )

    # Act
    catalog, diagnostics = runtime.create_scene_catalog(
        SceneCatalogRequest(
            representatives=(FrameCandidate("frame-a", b"image-a"),),
            selection_intent="ブログ画像を分類する",
        ),
        _resolved_model(ModelRole.SCENE_CATALOG),
        num_ctx=32768,
    )

    # Assert
    assert catalog.slugs[-1] == "other"
    assert diagnostics.attempt_count == 2
    assert diagnostics.validation_code == "scene_catalog_response_truncated"
    assert len(payloads) == 2
    second_prompt = _first_message(payloads[1])["content"]
    assert isinstance(second_prompt, str)
    assert "scene_catalog_response_truncated" in second_prompt


def test_repeated_truncated_response_fails_after_one_retry() -> None:
    """打ち切り応答が2回続いた場合にrunが停止されること。

    Arrange:
        - done reasonがlengthとなるschema-valid responseが用意される
    Act:
        - Scene Catalog推論が実行される
    Assert:
        - 2回だけrequestされresponse_truncatedで失敗すること
    """
    # Arrange
    calls = 0

    def requester(
        _method: str,
        _url: str,
        _payload: Mapping[str, object] | None,
        _timeout: float,
    ) -> object:
        nonlocal calls
        calls += 1
        response = _response(_catalog_payload())
        response["done_reason"] = "length"
        return response

    runtime = OllamaVisionRuntime(
        "http://localhost:11434",
        timeout_seconds=60.0,
        requester=requester,
        sleeper=lambda _seconds: None,
        model_state_resolver=_resolved_artifact,
    )

    # Act
    # Assert
    with pytest.raises(VisionRuntimeError) as captured:
        runtime.create_scene_catalog(
            SceneCatalogRequest(
                representatives=(FrameCandidate("frame-a", b"image-a"),),
                selection_intent="ブログ画像を分類する",
            ),
            _resolved_model(ModelRole.SCENE_CATALOG),
            num_ctx=32768,
        )
    assert calls == 2
    assert captured.value.reason is VisionRuntimeFailureReason.RESPONSE_INVALID
    assert captured.value.validation_code == "scene_catalog_response_truncated"


def test_candidate_domain_failure_stops_without_other_fallback() -> None:
    """Candidate Annotation domain failureが2回後にrun停止へ変換されること。

    Arrange:
        - 入力にないRepresentative Frameを2回返すAPIが用意される
    Act:
        - Candidate Annotation推論が実行される
    Assert:
        - domain_invalidで失敗しother fallbackが生成されないこと
    """
    # Arrange
    calls = 0

    def requester(
        _method: str,
        _url: str,
        _payload: Mapping[str, object] | None,
        _timeout: float,
    ) -> object:
        nonlocal calls
        calls += 1
        response = _annotation_payload()
        _first_frame_observation(response)["frame_id"] = "foreign-frame"
        return _response(response)

    runtime = OllamaVisionRuntime(
        "http://localhost:11434",
        timeout_seconds=60.0,
        requester=requester,
        sleeper=lambda _seconds: None,
        model_state_resolver=_resolved_artifact,
    )

    # Act
    # Assert
    with pytest.raises(VisionRuntimeError) as captured:
        runtime.annotate_candidate(
            _annotation_request(),
            _catalog(),
            _resolved_model(ModelRole.CANDIDATE_ANNOTATION),
            num_ctx=32768,
        )
    assert calls == 2
    assert captured.value.reason is VisionRuntimeFailureReason.DOMAIN_INVALID
    assert (
        captured.value.validation_code
        == "candidate_annotation_representative_frame_unknown"
    )
    assert "foreign-frame" not in str(captured.value)


def test_changed_ollama_tag_is_rejected_before_inference() -> None:
    """freeze後にOllama tagのdigestが変わった場合に推論前に停止されること。

    Arrange:
        - freeze済みidentityと異なるdigestを返すlocal model一覧が用意される
    Act:
        - Scene Catalog推論が実行される
    Assert:
        - chat requestが送られずmodel identity changedで失敗すること
    """
    # Arrange
    chat_calls = 0

    def requester(
        _method: str,
        url: str,
        _payload: Mapping[str, object] | None,
        _timeout: float,
    ) -> object:
        nonlocal chat_calls
        if url.endswith("/api/version"):
            return {"version": "0.31.2"}
        if url.endswith("/api/tags"):
            return {
                "models": [
                    {
                        "name": "qwen3-vl:8b-instruct",
                        "digest": "b" * 64,
                    }
                ]
            }
        chat_calls += 1
        return _response(_catalog_payload())

    runtime = OllamaVisionRuntime(
        "http://localhost:11434",
        timeout_seconds=60.0,
        requester=requester,
        sleeper=lambda _seconds: None,
    )

    # Act
    # Assert
    with pytest.raises(VisionRuntimeError) as captured:
        runtime.create_scene_catalog(
            SceneCatalogRequest(
                representatives=(FrameCandidate("frame-a", b"image-a"),),
                selection_intent="ブログ画像を分類する",
            ),
            _resolved_model(ModelRole.SCENE_CATALOG),
            num_ctx=32768,
        )
    assert chat_calls == 0
    assert captured.value.reason is VisionRuntimeFailureReason.MODEL_UNAVAILABLE
    assert captured.value.validation_code == "ollama_model_identity_changed"


def test_changed_ollama_runtime_is_rejected_before_inference() -> None:
    """freeze後にOllama runtime versionが変わった場合に推論前に停止されること。

    Arrange:
        - freeze済みruntimeと異なるserver versionが用意される
        - freeze済みdigestを指すlocal model一覧が用意される
    Act:
        - Scene Catalog推論が実行される
    Assert:
        - chat requestが送られずruntime identity changedで失敗すること
    """
    # Arrange
    chat_calls = 0

    def requester(
        _method: str,
        url: str,
        _payload: Mapping[str, object] | None,
        _timeout: float,
    ) -> object:
        nonlocal chat_calls
        if url.endswith("/api/version"):
            return {"version": "0.31.3"}
        if url.endswith("/api/tags"):
            return {
                "models": [
                    {
                        "name": "qwen3-vl:8b-instruct",
                        "digest": "a" * 64,
                    }
                ]
            }
        chat_calls += 1
        return _response(_catalog_payload())

    runtime = OllamaVisionRuntime(
        "http://localhost:11434",
        timeout_seconds=60.0,
        requester=requester,
        sleeper=lambda _seconds: None,
    )

    # Act
    # Assert
    with pytest.raises(VisionRuntimeError) as captured:
        runtime.create_scene_catalog(
            SceneCatalogRequest(
                representatives=(FrameCandidate("frame-a", b"image-a"),),
                selection_intent="ブログ画像を分類する",
            ),
            _resolved_model(ModelRole.SCENE_CATALOG),
            num_ctx=32768,
        )
    assert chat_calls == 0
    assert captured.value.reason is VisionRuntimeFailureReason.MODEL_UNAVAILABLE
    assert captured.value.validation_code == "ollama_runtime_identity_changed"


def test_changed_ollama_tag_is_rejected_after_inference() -> None:
    """推論中にOllama tagのdigestが変わった場合に応答が破棄されること。

    Arrange:
        - 推論前はfreeze済みdigest、推論後は異なるdigestが用意される
    Act:
        - Scene Catalog推論が実行される
    Assert:
        - chat応答が返ってもmodel identity changedで失敗すること
    """
    # Arrange
    tag_calls = 0
    chat_calls = 0

    def requester(
        _method: str,
        url: str,
        _payload: Mapping[str, object] | None,
        _timeout: float,
    ) -> object:
        nonlocal tag_calls, chat_calls
        if url.endswith("/api/version"):
            return {"version": "0.31.2"}
        if url.endswith("/api/tags"):
            tag_calls += 1
            fill = "a" if tag_calls == 1 else "b"
            return {
                "models": [
                    {
                        "name": "qwen3-vl:8b-instruct",
                        "digest": fill * 64,
                    }
                ]
            }
        chat_calls += 1
        return _response(_catalog_payload())

    runtime = OllamaVisionRuntime(
        "http://localhost:11434",
        timeout_seconds=60.0,
        requester=requester,
        sleeper=lambda _seconds: None,
    )

    # Act
    # Assert
    with pytest.raises(VisionRuntimeError) as captured:
        runtime.create_scene_catalog(
            SceneCatalogRequest(
                representatives=(FrameCandidate("frame-a", b"image-a"),),
                selection_intent="ブログ画像を分類する",
            ),
            _resolved_model(ModelRole.SCENE_CATALOG),
            num_ctx=32768,
        )
    assert tag_calls == 2
    assert chat_calls == 1
    assert captured.value.reason is VisionRuntimeFailureReason.MODEL_UNAVAILABLE
    assert captured.value.validation_code == "ollama_model_identity_changed"


def test_changed_ollama_runtime_is_rejected_after_inference() -> None:
    """推論中にOllama runtime versionが変わった場合に応答が破棄されること。

    Arrange:
        - 推論前はfreeze済みversion、推論後は異なるversionが用意される
        - 両確認でfreeze済みdigestを指すlocal model一覧が用意される
    Act:
        - Scene Catalog推論が実行される
    Assert:
        - chat応答が返ってもruntime identity changedで失敗すること
    """
    # Arrange
    version_calls = 0
    chat_calls = 0

    def requester(
        _method: str,
        url: str,
        _payload: Mapping[str, object] | None,
        _timeout: float,
    ) -> object:
        nonlocal version_calls, chat_calls
        if url.endswith("/api/version"):
            version_calls += 1
            version = "0.31.2" if version_calls == 1 else "0.31.3"
            return {"version": version}
        if url.endswith("/api/tags"):
            return {
                "models": [
                    {
                        "name": "qwen3-vl:8b-instruct",
                        "digest": "a" * 64,
                    }
                ]
            }
        chat_calls += 1
        return _response(_catalog_payload())

    runtime = OllamaVisionRuntime(
        "http://localhost:11434",
        timeout_seconds=60.0,
        requester=requester,
        sleeper=lambda _seconds: None,
    )

    # Act
    # Assert
    with pytest.raises(VisionRuntimeError) as captured:
        runtime.create_scene_catalog(
            SceneCatalogRequest(
                representatives=(FrameCandidate("frame-a", b"image-a"),),
                selection_intent="ブログ画像を分類する",
            ),
            _resolved_model(ModelRole.SCENE_CATALOG),
            num_ctx=32768,
        )
    assert version_calls == 2
    assert chat_calls == 1
    assert captured.value.reason is VisionRuntimeFailureReason.MODEL_UNAVAILABLE
    assert captured.value.validation_code == "ollama_runtime_identity_changed"


def test_tag_preflight_http_429_honors_http_date_retry_after() -> None:
    """推論前tag確認のHTTP 429でもHTTP-date待機が尊重されること。

    Arrange:
        - 初回tag確認に45秒後のHTTP-dateを返す429が用意される
        - 再試行時にfreeze済みdigestとvalid responseが用意される
    Act:
        - Scene Catalog推論が実行される
    Assert:
        - 待機が30秒へ制限され推論前tag確認から再試行されること
    """
    # Arrange
    tag_calls = 0
    sleeps: list[float] = []

    def requester(
        _method: str,
        url: str,
        _payload: Mapping[str, object] | None,
        _timeout: float,
    ) -> object:
        nonlocal tag_calls
        if url.endswith("/api/version"):
            return {"version": "0.31.2"}
        if url.endswith("/api/tags"):
            tag_calls += 1
            if tag_calls == 1:
                headers = Message()
                retry_at = datetime.now(timezone.utc) + timedelta(seconds=45)
                headers["Retry-After"] = format_datetime(retry_at, usegmt=True)
                raise HTTPError(url, 429, "rate limited", headers, None)
            return {
                "models": [
                    {
                        "name": "qwen3-vl:8b-instruct",
                        "digest": "a" * 64,
                    }
                ]
            }
        return _response(_catalog_payload())

    runtime = OllamaVisionRuntime(
        "http://localhost:11434",
        timeout_seconds=60.0,
        requester=requester,
        sleeper=sleeps.append,
    )

    # Act
    catalog, diagnostics = runtime.create_scene_catalog(
        SceneCatalogRequest(
            representatives=(FrameCandidate("frame-a", b"image-a"),),
            selection_intent="ブログ画像を分類する",
        ),
        _resolved_model(ModelRole.SCENE_CATALOG),
        num_ctx=32768,
    )

    # Assert
    assert catalog.slugs[-1] == "other"
    assert diagnostics.attempt_count == 2
    assert tag_calls == 3
    assert sleeps == [30.0]


@pytest.mark.parametrize(
    ("status_code", "expected_reason"),
    (
        (400, VisionRuntimeFailureReason.INVALID_REQUEST),
        (401, VisionRuntimeFailureReason.INVALID_REQUEST),
        (404, VisionRuntimeFailureReason.MODEL_UNAVAILABLE),
    ),
)
def test_tag_preflight_non_retryable_http_4xx_fails_immediately(
    status_code: int,
    expected_reason: VisionRuntimeFailureReason,
) -> None:
    """推論前tag確認の非retryable HTTP 4xxが即時fatalになること。

    Arrange:
        - 外部detailを含む非retryable HTTP 4xxがtag確認へ用意される
    Act:
        - Scene Catalog推論が実行される
    Assert:
        - 再試行されずstatusに対応する安全なfatal errorになること
    """
    # Arrange
    tag_calls = 0

    def requester(
        _method: str,
        url: str,
        _payload: Mapping[str, object] | None,
        _timeout: float,
    ) -> object:
        nonlocal tag_calls
        if url.endswith("/api/version"):
            return {"version": "0.31.2"}
        tag_calls += 1
        raise HTTPError(
            url,
            status_code,
            "token-secret /private/model",
            hdrs=Message(),
            fp=None,
        )

    runtime = OllamaVisionRuntime(
        "http://localhost:11434",
        timeout_seconds=60.0,
        requester=requester,
        sleeper=lambda _seconds: None,
    )

    # Act
    # Assert
    with pytest.raises(VisionRuntimeError) as captured:
        runtime.create_scene_catalog(
            SceneCatalogRequest(
                representatives=(FrameCandidate("frame-a", b"image-a"),),
                selection_intent="ブログ画像を分類する",
            ),
            _resolved_model(ModelRole.SCENE_CATALOG),
            num_ctx=32768,
        )
    assert tag_calls == 1
    assert captured.value.reason is expected_reason
    assert "token-secret" not in str(captured.value)
    assert "/private/model" not in str(captured.value)


def test_invalid_tag_preflight_retries_without_repairing_prompt() -> None:
    """不正なtag応答後の再試行でmodel promptが変更されないこと。

    Arrange:
        - 初回だけ不正なlocal model一覧を返すAPIが用意される
        - 再試行時にfreeze済みdigestとvalid responseが用意される
    Act:
        - Scene Catalog推論が実行される
    Assert:
        - 診断codeは保持され、未実行modelのpromptへ修復指示が入らないこと
    """
    # Arrange
    tag_calls = 0
    chat_payloads: list[Mapping[str, object]] = []

    def requester(
        _method: str,
        url: str,
        payload: Mapping[str, object] | None,
        _timeout: float,
    ) -> object:
        nonlocal tag_calls
        if url.endswith("/api/version"):
            return {"version": "0.31.2"}
        if url.endswith("/api/tags"):
            tag_calls += 1
            if tag_calls == 1:
                return {"models": "invalid"}
            return {
                "models": [
                    {
                        "name": "qwen3-vl:8b-instruct",
                        "digest": "a" * 64,
                    }
                ]
            }
        assert payload is not None
        chat_payloads.append(payload)
        return _response(_catalog_payload())

    runtime = OllamaVisionRuntime(
        "http://localhost:11434",
        timeout_seconds=60.0,
        requester=requester,
        sleeper=lambda _seconds: None,
    )

    # Act
    _, diagnostics = runtime.create_scene_catalog(
        SceneCatalogRequest(
            representatives=(FrameCandidate("frame-a", b"image-a"),),
            selection_intent="ブログ画像を分類する",
        ),
        _resolved_model(ModelRole.SCENE_CATALOG),
        num_ctx=32768,
    )

    # Assert
    assert diagnostics.attempt_count == 2
    assert diagnostics.validation_code == "ollama_model_identity_response_invalid"
    assert tag_calls == 3
    assert len(chat_payloads) == 1
    assert "validation_code" not in str(_first_message(chat_payloads[0])["content"])


def test_candidate_without_context_is_explicitly_unavailable() -> None:
    """Context CueなしのCandidateがunavailableとして評価されること。

    Arrange:
        - Context Cueを持たないCandidate Momentとunavailable responseが用意される
    Act:
        - Candidate Annotation推論が実行される
    Assert:
        - supporting Cueなしのunavailableが正常なannotationになること
    """
    # Arrange
    request = _annotation_request()
    request_without_context = CandidateAnnotationRequest(
        moment=request.moment,
        frame_candidates=request.frame_candidates,
        context_cues=(),
        video_set_progress=request.video_set_progress,
        selection_intent=request.selection_intent,
        cue_selection_policy_version=request.cue_selection_policy_version,
    )
    response = _annotation_payload()
    response["context_relevance"] = "unavailable"
    response["supporting_context_cue_ids"] = []
    payloads: list[Mapping[str, object]] = []

    def requester(
        _method: str,
        _url: str,
        payload: Mapping[str, object] | None,
        _timeout: float,
    ) -> object:
        assert payload is not None
        payloads.append(payload)
        return _response(response)

    runtime = OllamaVisionRuntime(
        "http://localhost:11434",
        timeout_seconds=60.0,
        requester=requester,
        sleeper=lambda _seconds: None,
        model_state_resolver=_resolved_artifact,
    )

    # Act
    annotation, diagnostics = runtime.annotate_candidate(
        request_without_context,
        _catalog(),
        _resolved_model(ModelRole.CANDIDATE_ANNOTATION),
        num_ctx=32768,
    )

    # Assert
    assert annotation.context_relevance == "unavailable"
    assert annotation.supporting_context_cue_ids == ()
    assert diagnostics.context_cue_count == 0
    schema = payloads[0]["format"]
    assert isinstance(schema, dict)
    properties = schema["properties"]
    assert isinstance(properties, dict)
    assert properties["context_relevance"]["enum"] == ["unavailable"]
    assert properties["supporting_context_cue_ids"]["maxItems"] == 0


def test_candidate_schema_limits_references_to_request_members() -> None:
    """Candidateの参照先がrequest内のIDへschemaで限定されること。

    Arrange:
        - frame、Catalog、Context Cueを持つCandidate requestが用意される
    Act:
        - Candidate Annotation推論が実行される
    Assert:
        - frame、scene、Cue IDとCueありrelevanceが入力集合へ限定されること
    """
    # Arrange
    payloads: list[Mapping[str, object]] = []

    def requester(
        _method: str,
        _url: str,
        payload: Mapping[str, object] | None,
        _timeout: float,
    ) -> object:
        assert payload is not None
        payloads.append(payload)
        return _response(_annotation_payload())

    runtime = OllamaVisionRuntime(
        "http://localhost:11434",
        timeout_seconds=60.0,
        requester=requester,
        sleeper=lambda _seconds: None,
        model_state_resolver=_resolved_artifact,
    )

    # Act
    runtime.annotate_candidate(
        _annotation_request(),
        _catalog(),
        _resolved_model(ModelRole.CANDIDATE_ANNOTATION),
        num_ctx=32768,
    )

    # Assert
    schema = payloads[0]["format"]
    assert isinstance(schema, dict)
    properties = schema["properties"]
    assert isinstance(properties, dict)
    observations = properties["frame_observations"]
    assert observations["minItems"] == 1
    assert observations["maxItems"] == 1
    observation_properties = observations["items"]["properties"]
    assert observation_properties["frame_id"]["enum"] == ["frame-a"]
    assert observation_properties["scene_slug"]["enum"] == [
        "exploration",
        "battle",
        "other",
    ]
    assert observation_properties["interface_kind"]["enum"] == [
        "none",
        "document",
        "shop",
        "map",
        "save",
        "tutorial_help",
        "other_interface",
        "title",
    ]
    assert observation_properties["prominent_event_portrait"] == {"type": "boolean"}
    assert observation_properties["visible_dialogue_text"] == {"type": "boolean"}
    assert observation_properties["visible_action"] == {"type": "boolean"}
    assert observation_properties["visible_character_or_enemy"] == {"type": "boolean"}
    assert observation_properties["combat_action"] == {"type": "boolean"}
    assert observation_properties["visible_player_character"] == {"type": "boolean"}
    assert observation_properties["visible_opponent"] == {"type": "boolean"}
    assert properties["context_relevance"]["enum"] == ["none", "weak", "strong"]
    cue_schema = properties["supporting_context_cue_ids"]
    assert cue_schema["items"]["enum"] == ["cue-a"]
    assert cue_schema["maxItems"] == 1


def test_candidate_prompt_defines_blog_usefulness_boundaries() -> None:
    """Candidate promptへブログ用途の意味境界が明示されること。

    Arrange:
        - frame、Catalog、Context Cueを持つCandidate requestが用意される
    Act:
        - Candidate Annotation推論が実行される
    Assert:
        - 代表frame、Blog Image Type、説明価値、Context、Spoilerの基準が送信されること
    """
    # Arrange
    payloads: list[Mapping[str, object]] = []

    def requester(
        _method: str,
        _url: str,
        payload: Mapping[str, object] | None,
        _timeout: float,
    ) -> object:
        assert payload is not None
        payloads.append(payload)
        return _response(_annotation_payload())

    runtime = OllamaVisionRuntime(
        "http://localhost:11434",
        timeout_seconds=60.0,
        requester=requester,
        sleeper=lambda _seconds: None,
        model_state_resolver=_resolved_artifact,
    )

    # Act
    runtime.annotate_candidate(
        _annotation_request(),
        _catalog(),
        _resolved_model(ModelRole.CANDIDATE_ANNOTATION),
        num_ctx=32768,
    )

    # Assert
    prompt = _last_message(payloads[0])["content"]
    assert isinstance(prompt, str)
    assert "各frameを他のframeの内容と混ぜず" in prompt
    assert "大きな発光やエフェクトで主対象が隠れる" in prompt
    assert "gameplay_action=" in prompt
    assert "tutorial_help" in prompt
    assert "visible_dialogue_text" in prompt
    assert "visible_action" in prompt
    assert "visible_character_or_enemy" in prompt
    assert "combat_action" in prompt
    assert "visible_player_character" in prompt
    assert "visible_opponent" in prompt
    assert "人物portraitだけ" in prompt
    assert "手紙・手記・日誌・記録" in prompt
    assert "prominent_event_portrait" in prompt
    assert "画面隅の小さな" in prompt
    assert "explanation_valueのnone=" in prompt
    assert "context_relevanceのstrong=" in prompt
    assert "spoiler_riskのhigh=" in prompt


def test_candidate_frames_are_labeled_and_selected_from_per_frame_observations() -> (
    None
):
    """画像とIDが対応付けられ、frame別観測から代表画像が決定されること。

    Arrange:
        - shop、待機画面、台詞eventの3枚とframe別応答が用意される
    Act:
        - Candidate Annotation推論が実行される
    Assert:
        - 各画像が対応するIDだけを持つmessageで送信されること
        - 同じhigh評価では台詞eventのframeが代表として構築されること
    """
    # Arrange
    payloads: list[Mapping[str, object]] = []
    request = _annotation_request_with_frame_ids(("frame-a", "frame-b", "frame-c"))

    def requester(
        _method: str,
        _url: str,
        payload: Mapping[str, object] | None,
        _timeout: float,
    ) -> object:
        assert payload is not None
        payloads.append(payload)
        return _response(
            _frame_observation_payload(
                (
                    ("frame-a", "exploration", "shop", "high", "menu"),
                    (
                        "frame-b",
                        "exploration",
                        "gameplay_idle",
                        "high",
                        "none",
                    ),
                    (
                        "frame-c",
                        "battle",
                        "event_dialogue",
                        "high",
                        "dialogue",
                    ),
                )
            )
        )

    runtime = OllamaVisionRuntime(
        "http://localhost:11434",
        timeout_seconds=60.0,
        requester=requester,
        sleeper=lambda _seconds: None,
        model_state_resolver=_resolved_artifact,
    )

    # Act
    annotation, _diagnostics = runtime.annotate_candidate(
        request,
        _catalog(),
        _resolved_model(ModelRole.CANDIDATE_ANNOTATION),
        num_ctx=32768,
    )

    # Assert
    messages = payloads[0]["messages"]
    assert isinstance(messages, list)
    frame_message_contents = [message["content"] for message in messages[:-1]]
    frame_message_ids = [
        content.split("。", maxsplit=1)[0] for content in frame_message_contents
    ]
    assert frame_message_ids == [
        "frame_candidate_id=frame-a",
        "frame_candidate_id=frame-b",
        "frame_candidate_id=frame-c",
    ]
    assert all(
        "この画像だけに実際に見えるもの" in content
        for content in frame_message_contents
    )
    assert all("人物portraitだけ" in content for content in frame_message_contents)
    assert [message["images"] for message in messages[:-1]] == [
        ["aW1hZ2UtZnJhbWUtYQ=="],
        ["aW1hZ2UtZnJhbWUtYg=="],
        ["aW1hZ2UtZnJhbWUtYw=="],
    ]
    assert "images" not in messages[-1]
    assert annotation.candidate.identifier == "frame-c"
    assert annotation.blog_image_type == "event"
    assert annotation.explanation_value == "high"
    assert annotation.screen_text_kind == "dialogue"


def test_tutorial_observation_is_not_eligible_for_selection_filling() -> None:
    """modelがmediumと返してもtutorialが説明価値なしへ正規化されること。

    Arrange:
        - Context Cueなしのtutorial_help frameとmedium応答が用意される
    Act:
        - Candidate Annotation推論が実行される
    Assert:
        - menu分類を保ちながら説明価値なしへ正規化されること
    """
    # Arrange
    request = _annotation_request_with_frame_ids(("frame-a",), include_context=False)
    runtime = OllamaVisionRuntime(
        "http://localhost:11434",
        timeout_seconds=60.0,
        requester=lambda _method, _url, _payload, _timeout: _response(
            _frame_observation_payload(
                (("frame-a", "exploration", "tutorial_help", "medium", "dialogue"),),
                context_relevance="unavailable",
                supporting_context_cue_ids=(),
            )
        ),
        sleeper=lambda _seconds: None,
        model_state_resolver=_resolved_artifact,
    )

    # Act
    annotation, _diagnostics = runtime.annotate_candidate(
        request,
        _catalog(),
        _resolved_model(ModelRole.CANDIDATE_ANNOTATION),
        num_ctx=32768,
    )

    # Assert
    assert annotation.blog_image_type == "menu"
    assert annotation.explanation_value == "none"
    assert annotation.screen_text_kind == "menu"


def test_atomic_observations_override_ambiguous_tutorial_content() -> None:
    """単純観測でtutorialと判明したframeがmodelのevent分類を上書きすること。

    Arrange:
        - event_dialogueかつhighだが、静止tutorialを示すatomic観測が用意される
    Act:
        - Candidate Annotation推論が実行される
    Assert:
        - menu分類の説明価値なしへ決定的に正規化されること
    """
    # Arrange
    response = _frame_observation_payload(
        (("frame-a", "exploration", "event_dialogue", "high", "dialogue"),)
    )
    observation = _first_frame_observation(response)
    observation.update(
        {
            "interface_kind": "tutorial_help",
            "visible_dialogue_text": True,
            "visible_action": False,
            "visible_character_or_enemy": False,
        }
    )
    runtime = OllamaVisionRuntime(
        "http://localhost:11434",
        timeout_seconds=60.0,
        requester=lambda _method, _url, _payload, _timeout: _response(response),
        sleeper=lambda _seconds: None,
        model_state_resolver=_resolved_artifact,
    )

    # Act
    annotation, _diagnostics = runtime.annotate_candidate(
        _annotation_request(),
        _catalog(),
        _resolved_model(ModelRole.CANDIDATE_ANNOTATION),
        num_ctx=32768,
    )

    # Assert
    assert annotation.blog_image_type == "menu"
    assert annotation.explanation_value == "none"
    assert annotation.screen_text_kind == "menu"


@pytest.mark.parametrize(
    (
        "content_kind",
        "interface_kind",
        "prominent_event_portrait",
        "expected_blog_image_type",
    ),
    (
        ("other", "document", False, "menu"),
        ("gameplay_idle", "none", True, "event"),
    ),
)
def test_static_document_and_silent_event_portrait_are_not_eligible(
    content_kind: str,
    interface_kind: str,
    prominent_event_portrait: bool,
    expected_blog_image_type: str,
) -> None:
    """静止文書と台詞のないイベント立ち絵が掲載不可にされること。

    Arrange:
        - 高評価だが静止文書または台詞のないイベント立ち絵の応答が用意される
    Act:
        - Candidate Annotation推論が実行される
    Assert:
        - 画面種別を保ちながら説明価値なしへ正規化されること
    """
    # Arrange
    response = _frame_observation_payload(
        (("frame-a", "exploration", content_kind, "high", "none"),)
    )
    observation = _first_frame_observation(response)
    observation.update(
        {
            "interface_kind": interface_kind,
            "prominent_event_portrait": prominent_event_portrait,
            "visible_dialogue_text": False,
            "visible_action": False,
        }
    )
    runtime = OllamaVisionRuntime(
        "http://localhost:11434",
        timeout_seconds=60.0,
        requester=lambda _method, _url, _payload, _timeout: _response(response),
        sleeper=lambda _seconds: None,
        model_state_resolver=_resolved_artifact,
    )

    # Act
    annotation, _diagnostics = runtime.annotate_candidate(
        _annotation_request(),
        _catalog(),
        _resolved_model(ModelRole.CANDIDATE_ANNOTATION),
        num_ctx=32768,
    )

    # Assert
    assert annotation.blog_image_type == expected_blog_image_type
    assert annotation.explanation_value == "none"


def test_combat_without_both_visible_participants_has_no_explanation_value() -> None:
    """戦闘の片方が見えないframeがmodelの高評価より優先して除外されること。

    Arrange:
        - 戦闘中だが発光でplayerが判別できないhigh応答が用意される
    Act:
        - Candidate Annotation推論が実行される
    Assert:
        - 戦闘分類を保ちながら説明価値なしへ正規化されること
    """
    # Arrange
    response = _frame_observation_payload(
        (("frame-a", "battle", "event_action", "high", "hud"),)
    )
    observation = _first_frame_observation(response)
    observation.update(
        {
            "combat_action": True,
            "visible_player_character": False,
            "visible_opponent": True,
        }
    )
    runtime = OllamaVisionRuntime(
        "http://localhost:11434",
        timeout_seconds=60.0,
        requester=lambda _method, _url, _payload, _timeout: _response(response),
        sleeper=lambda _seconds: None,
        model_state_resolver=_resolved_artifact,
    )

    # Act
    annotation, _diagnostics = runtime.annotate_candidate(
        _annotation_request(),
        _catalog(),
        _resolved_model(ModelRole.CANDIDATE_ANNOTATION),
        num_ctx=32768,
    )

    # Assert
    assert annotation.blog_image_type == "event"
    assert annotation.explanation_value == "none"


def test_candidate_relationship_failure_is_repaired_with_explicit_contract() -> None:
    """Cueなし応答の関係違反が明示契約と個別codeで修復されること。

    Arrange:
        - Cueなし入力と、初回だけrelevanceとspoiler関係が不正な応答が用意される
    Act:
        - Candidate Annotation推論が実行される
    Assert:
        - 条件付き契約と個別validation codeを受けた再試行が成功すること
    """
    # Arrange
    request = _annotation_request()
    request_without_context = CandidateAnnotationRequest(
        moment=request.moment,
        frame_candidates=request.frame_candidates,
        context_cues=(),
        video_set_progress=request.video_set_progress,
        selection_intent=request.selection_intent,
        cue_selection_policy_version=request.cue_selection_policy_version,
    )
    payloads: list[Mapping[str, object]] = []

    def requester(
        _method: str,
        _url: str,
        payload: Mapping[str, object] | None,
        _timeout: float,
    ) -> object:
        assert payload is not None
        payloads.append(payload)
        response = _annotation_payload()
        response["context_relevance"] = "unavailable"
        response["supporting_context_cue_ids"] = []
        observation = _first_frame_observation(response)
        observation["spoiler_risk"] = "none"
        observation["spoiler_evidence"] = "画面由来の根拠" if len(payloads) == 1 else ""
        return _response(response)

    runtime = OllamaVisionRuntime(
        "http://localhost:11434",
        timeout_seconds=60.0,
        requester=requester,
        sleeper=lambda _seconds: None,
        model_state_resolver=_resolved_artifact,
    )

    # Act
    annotation, diagnostics = runtime.annotate_candidate(
        request_without_context,
        _catalog(),
        _resolved_model(ModelRole.CANDIDATE_ANNOTATION),
        num_ctx=32768,
    )

    # Assert
    assert annotation.context_relevance == "unavailable"
    assert diagnostics.attempt_count == 2
    assert diagnostics.validation_code == "candidate_annotation_relationship_invalid"
    first_prompt = _last_message(payloads[0])["content"]
    second_prompt = _last_message(payloads[1])["content"]
    assert isinstance(first_prompt, str)
    assert isinstance(second_prompt, str)
    assert "context_cuesが空ならcontext_relevanceはunavailable" in first_prompt
    assert "spoiler_riskがnoneならspoiler_evidenceは空文字列" in first_prompt
    assert "candidate_annotation_relationship_invalid" in second_prompt
    assert "関係を必ず修正します" in second_prompt
    assert "low・medium・highならspoiler_evidenceは" in second_prompt


def test_candidate_without_context_rejects_none_relevance() -> None:
    """Context Cueなしでnone relevanceが返された場合に拒否されること。

    Arrange:
        - Context CueなしのCandidateとnone responseが用意される
    Act:
        - Candidate Annotation推論が実行される
    Assert:
        - unavailable以外がdomain invalidとして拒否されること
    """
    # Arrange
    request = _annotation_request()
    request_without_context = CandidateAnnotationRequest(
        moment=request.moment,
        frame_candidates=request.frame_candidates,
        context_cues=(),
        video_set_progress=request.video_set_progress,
        selection_intent=request.selection_intent,
        cue_selection_policy_version=request.cue_selection_policy_version,
    )
    response = _annotation_payload()
    response["context_relevance"] = "none"
    response["supporting_context_cue_ids"] = []
    runtime = OllamaVisionRuntime(
        "http://localhost:11434",
        timeout_seconds=60.0,
        requester=lambda _method, _url, _payload, _timeout: _response(response),
        sleeper=lambda _seconds: None,
        model_state_resolver=_resolved_artifact,
    )

    # Act
    # Assert
    with pytest.raises(VisionRuntimeError) as captured:
        runtime.annotate_candidate(
            request_without_context,
            _catalog(),
            _resolved_model(ModelRole.CANDIDATE_ANNOTATION),
            num_ctx=32768,
        )
    assert captured.value.reason is VisionRuntimeFailureReason.DOMAIN_INVALID
    assert captured.value.validation_code == "candidate_annotation_context_invalid"


def test_candidate_with_context_rejects_unavailable_relevance() -> None:
    """Context Cueありでunavailable relevanceが返された場合に拒否されること。

    Arrange:
        - Context CueありのCandidateとunavailable responseが用意される
    Act:
        - Candidate Annotation推論が実行される
    Assert:
        - Context Cueを評価しない応答がdomain invalidとして拒否されること
    """
    # Arrange
    response = _annotation_payload()
    response["context_relevance"] = "unavailable"
    response["supporting_context_cue_ids"] = []
    runtime = OllamaVisionRuntime(
        "http://localhost:11434",
        timeout_seconds=60.0,
        requester=lambda _method, _url, _payload, _timeout: _response(response),
        sleeper=lambda _seconds: None,
        model_state_resolver=_resolved_artifact,
    )

    # Act
    # Assert
    with pytest.raises(VisionRuntimeError) as captured:
        runtime.annotate_candidate(
            _annotation_request(),
            _catalog(),
            _resolved_model(ModelRole.CANDIDATE_ANNOTATION),
            num_ctx=32768,
        )
    assert captured.value.reason is VisionRuntimeFailureReason.DOMAIN_INVALID
    assert captured.value.validation_code == "candidate_annotation_context_invalid"


def test_candidate_redacts_verbatim_context_cue_in_spoiler_evidence() -> None:
    """Context Cue本文が自由文fieldへ逐語再出力された場合に安全化されること。

    Arrange:
        - Context Cue本文をspoiler evidenceへそのまま返すresponseが用意される
    Act:
        - Candidate Annotation推論が実行される
    Assert:
        - raw textを含むevidenceが非逐語表現へ置換されること
    """
    # Arrange
    request = _annotation_request()
    response = _annotation_payload()
    _first_frame_observation(response)["spoiler_evidence"] = request.context_cues[
        0
    ].text
    payloads: list[Mapping[str, object]] = []

    def requester(
        _method: str,
        _url: str,
        payload: Mapping[str, object] | None,
        _timeout: float,
    ) -> object:
        assert payload is not None
        payloads.append(payload)
        return _response(response)

    runtime = OllamaVisionRuntime(
        "http://localhost:11434",
        timeout_seconds=60.0,
        requester=requester,
        sleeper=lambda _seconds: None,
        model_state_resolver=_resolved_artifact,
    )

    # Act
    annotation, diagnostics = runtime.annotate_candidate(
        request,
        _catalog(),
        _resolved_model(ModelRole.CANDIDATE_ANNOTATION),
        num_ctx=32768,
    )

    # Assert
    free_text = (
        annotation.summary,
        annotation.frame_choice_reason or "",
        annotation.spoiler_evidence,
    )
    assert candidate_annotation_free_text_is_safe(
        free_text,
        tuple(item.text for item in request.context_cues),
    )
    assert request.context_cues[0].text not in free_text
    assert diagnostics.validation_code == (
        "candidate_annotation_verbatim_context_redacted"
    )
    assert diagnostics.attempt_count == 1
    assert len(payloads) == 1
    first_prompt = _last_message(payloads[0])["content"]
    assert isinstance(first_prompt, str)
    assert "正規化後3〜5文字のCueは全文" in first_prompt


def test_candidate_normalizes_multiline_spoiler_evidence_for_publication() -> None:
    """複数行のSpoiler Evidenceが公開可能な1行へ正規化されること。

    Arrange:
        - 画像由来だが改行を含むSpoiler Evidence応答が用意される
    Act:
        - Candidate Annotation推論が実行される
    Assert:
        - 内容を保った1行の安全なEvidenceが返されること
    """
    # Arrange
    response = _annotation_payload()
    observation = _first_frame_observation(response)
    observation["spoiler_evidence"] = (
        "重要人物の姿が画面で示される\n次の形態が表示される"
    )
    runtime = OllamaVisionRuntime(
        "http://localhost:11434",
        timeout_seconds=60.0,
        requester=lambda _method, _url, _payload, _timeout: _response(response),
        sleeper=lambda _seconds: None,
        model_state_resolver=_resolved_artifact,
    )

    # Act
    annotation, _diagnostics = runtime.annotate_candidate(
        _annotation_request(),
        _catalog(),
        _resolved_model(ModelRole.CANDIDATE_ANNOTATION),
        num_ctx=32768,
    )

    # Assert
    assert annotation.spoiler_evidence == (
        "重要人物の姿が画面で示される 次の形態が表示される"
    )
    assert not string_looks_private(annotation.spoiler_evidence)


@pytest.mark.parametrize(
    "unsafe_evidence",
    (
        "/private/model/result",
        "https://example.invalid/model/result",
    ),
)
def test_candidate_replaces_private_looking_spoiler_evidence(
    unsafe_evidence: str,
) -> None:
    """非公開形式のSpoiler Evidenceが決定的な安全文へ置換されること。

    Arrange:
        - 絶対pathまたはendpoint形式のSpoiler Evidence応答が用意される
    Act:
        - Candidate Annotation推論が実行される
    Assert:
        - raw応答を含まない安全なEvidenceへ置換されること
    """
    # Arrange
    response = _annotation_payload()
    _first_frame_observation(response)["spoiler_evidence"] = unsafe_evidence
    runtime = OllamaVisionRuntime(
        "http://localhost:11434",
        timeout_seconds=60.0,
        requester=lambda _method, _url, _payload, _timeout: _response(response),
        sleeper=lambda _seconds: None,
        model_state_resolver=_resolved_artifact,
    )

    # Act
    annotation, _diagnostics = runtime.annotate_candidate(
        _annotation_request(),
        _catalog(),
        _resolved_model(ModelRole.CANDIDATE_ANNOTATION),
        num_ctx=32768,
    )

    # Assert
    assert annotation.spoiler_evidence == "high相当の進行情報を映像から判定"
    assert unsafe_evidence not in annotation.spoiler_evidence
    assert not string_looks_private(annotation.spoiler_evidence)


@pytest.mark.parametrize(
    ("cue_text", "leaked_text"),
    (
        ("犯人A", "犯人Aが判明"),
        ("「正体は王だ」", "正体は王だ"),
        ("王都は陥落した。次の目的地は北の塔だ。", "次の目的地は北の塔だ"),
    ),
)
def test_candidate_redacts_normalized_or_partial_context_cue_quote(
    cue_text: str,
    leaked_text: str,
) -> None:
    """Context Cueの引用符除去または一部引用が安全化されること。

    Arrange:
        - 引用符付きまたは複数文のContext Cueが用意される
        - Cueの正規化後全文または一文だけを返すresponseが用意される
    Act:
        - Candidate Annotation推論が実行される
    Assert:
        - 十分長い逐語spanを含むfieldが非逐語表現へ置換されること
    """
    # Arrange
    request = _annotation_request_with_context_text(cue_text)
    response = _annotation_payload()
    _first_frame_observation(response)["spoiler_evidence"] = leaked_text
    runtime = OllamaVisionRuntime(
        "http://localhost:11434",
        timeout_seconds=60.0,
        requester=lambda _method, _url, _payload, _timeout: _response(response),
        sleeper=lambda _seconds: None,
        model_state_resolver=_resolved_artifact,
    )

    # Act
    annotation, diagnostics = runtime.annotate_candidate(
        request,
        _catalog(),
        _resolved_model(ModelRole.CANDIDATE_ANNOTATION),
        num_ctx=32768,
    )

    # Assert
    assert annotation.spoiler_evidence != leaked_text
    assert candidate_annotation_free_text_is_safe(
        (annotation.spoiler_evidence,),
        (cue_text,),
    )
    assert diagnostics.validation_code == (
        "candidate_annotation_verbatim_context_redacted"
    )


def test_candidate_uses_symbol_omission_when_safe_fallback_matches_context() -> None:
    """安全化fallbackもCueと一致する場合に記号だけへ置換されること。

    Arrange:
        - Scene由来fallbackと一致するContext Cueと逐語responseが用意される
    Act:
        - Candidate Annotation推論が実行される
    Assert:
        - Cue文字を持たない省略記号がsummaryへ返されること
    """
    # Arrange
    cue_text = "戦闘の台詞のあるイベント。戦闘に分類されるeventの場面"
    request = _annotation_request_with_context_text(cue_text)
    response = _annotation_payload()
    runtime = OllamaVisionRuntime(
        "http://localhost:11434",
        timeout_seconds=60.0,
        requester=lambda _method, _url, _payload, _timeout: _response(response),
        sleeper=lambda _seconds: None,
        model_state_resolver=_resolved_artifact,
    )

    # Act
    annotation, diagnostics = runtime.annotate_candidate(
        request,
        _catalog(),
        _resolved_model(ModelRole.CANDIDATE_ANNOTATION),
        num_ctx=32768,
    )

    # Assert
    assert annotation.summary == "［…］"
    assert candidate_annotation_free_text_is_safe(
        (annotation.summary,),
        (cue_text,),
    )
    assert diagnostics.validation_code == (
        "candidate_annotation_verbatim_context_redacted"
    )


@pytest.mark.parametrize(
    ("cue_text", "annotation_summary"),
    (
        ("はい", "人物がはいと返事する場面"),
        ("OK", "画面にOK表示が示される場面"),
    ),
)
def test_candidate_allows_ambiguous_one_or_two_character_cue_occurrence(
    cue_text: str,
    annotation_summary: str,
) -> None:
    """一般的な1〜2文字Cueの出現が逐語引用と判定されないこと。

    Arrange:
        - 一般的な1〜2文字Cueと、その文字列を含む要約が用意される
    Act:
        - Candidate Annotation推論が実行される
    Assert:
        - 独立生成との区別がつかない短い一致が受理されること
    """
    # Arrange
    request = _annotation_request_with_context_text(cue_text)
    response = _annotation_payload()
    catalog = _catalog_with_battle_display(annotation_summary)
    runtime = OllamaVisionRuntime(
        "http://localhost:11434",
        timeout_seconds=60.0,
        requester=lambda _method, _url, _payload, _timeout: _response(response),
        sleeper=lambda _seconds: None,
        model_state_resolver=_resolved_artifact,
    )

    # Act
    annotation, _ = runtime.annotate_candidate(
        request,
        catalog,
        _resolved_model(ModelRole.CANDIDATE_ANNOTATION),
        num_ctx=32768,
    )

    # Assert
    assert annotation.summary.startswith(annotation_summary)


def test_retryable_transport_failure_is_retried_with_same_semantic_input() -> None:
    """一時transport failureが同じsemantic入力で一度だけ再試行されること。

    Arrange:
        - 初回timeout後にhigh Spoiler Riskを返すAPIが用意される
    Act:
        - Candidate Annotation推論が実行される
    Assert:
        - 2回目が成功しmajor spoiler evidenceが保持されること
    """
    # Arrange
    payloads: list[Mapping[str, object]] = []

    def requester(
        _method: str,
        _url: str,
        payload: Mapping[str, object] | None,
        _timeout: float,
    ) -> object:
        assert payload is not None
        payloads.append(payload)
        if len(payloads) == 1:
            raise TimeoutError("token-secret /private/path")
        return _response(_annotation_payload())

    runtime = OllamaVisionRuntime(
        "http://localhost:11434",
        timeout_seconds=60.0,
        requester=requester,
        sleeper=lambda _seconds: None,
        model_state_resolver=_resolved_artifact,
    )

    # Act
    annotation, diagnostics = runtime.annotate_candidate(
        _annotation_request(),
        _catalog(),
        _resolved_model(ModelRole.CANDIDATE_ANNOTATION),
        num_ctx=32768,
    )

    # Assert
    assert diagnostics.attempt_count == 2
    assert annotation.spoiler_risk == "high"
    assert annotation.spoiler_evidence == "最終ボスの正体が画面で明示される"
    assert (
        _first_message(payloads[0])["images"] == _first_message(payloads[1])["images"]
    )
    assert (
        _first_message(payloads[0])["content"] == _first_message(payloads[1])["content"]
    )
    assert "ollama_transport_failure" not in str(_first_message(payloads[1])["content"])


def test_non_retryable_http_4xx_fails_immediately_without_external_detail() -> None:
    """408/429以外のHTTP 4xxが安全なfatal errorになること。

    Arrange:
        - credentialを含むHTTP 400 responseが用意される
    Act:
        - Scene Catalog推論が実行される
    Assert:
        - 再試行されず外部detailがerrorへ出ないこと
    """
    # Arrange
    calls = 0

    def requester(
        _method: str,
        url: str,
        _payload: Mapping[str, object] | None,
        _timeout: float,
    ) -> object:
        nonlocal calls
        calls += 1
        raise HTTPError(
            url,
            400,
            "token-secret /private/model",
            hdrs=Message(),
            fp=None,
        )

    runtime = OllamaVisionRuntime(
        "http://localhost:11434",
        timeout_seconds=60.0,
        requester=requester,
        sleeper=lambda _seconds: None,
        model_state_resolver=_resolved_artifact,
    )

    # Act
    # Assert
    with pytest.raises(VisionRuntimeError) as captured:
        runtime.create_scene_catalog(
            SceneCatalogRequest(
                representatives=(FrameCandidate("frame-a", b"image-a"),),
                selection_intent="ブログ画像を分類する",
            ),
            _resolved_model(ModelRole.SCENE_CATALOG),
            num_ctx=32768,
        )
    assert calls == 1
    assert captured.value.reason is VisionRuntimeFailureReason.INVALID_REQUEST
    assert "token-secret" not in str(captured.value)
    assert "/private/model" not in str(captured.value)


def test_http_429_honors_capped_retry_after() -> None:
    """HTTP 429のRetry-Afterが最大30秒まで尊重されること。

    Arrange:
        - 初回にRetry-After 45秒の429、2回目にvalid responseが用意される
    Act:
        - Scene Catalog推論が実行される
    Assert:
        - 待機が30秒へ制限され同じrequestが一度だけ再試行されること
    """
    # Arrange
    calls = 0
    sleeps: list[float] = []

    def requester(
        _method: str,
        url: str,
        _payload: Mapping[str, object] | None,
        _timeout: float,
    ) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            headers = Message()
            headers["Retry-After"] = "45"
            raise HTTPError(url, 429, "rate limited", headers, None)
        return _response(_catalog_payload())

    runtime = OllamaVisionRuntime(
        "http://localhost:11434",
        timeout_seconds=60.0,
        requester=requester,
        sleeper=sleeps.append,
        model_state_resolver=_resolved_artifact,
    )

    # Act
    catalog, diagnostics = runtime.create_scene_catalog(
        SceneCatalogRequest(
            representatives=(FrameCandidate("frame-a", b"image-a"),),
            selection_intent="ブログ画像を分類する",
        ),
        _resolved_model(ModelRole.SCENE_CATALOG),
        num_ctx=32768,
    )

    # Assert
    assert catalog.slugs[-1] == "other"
    assert diagnostics.attempt_count == 2
    assert calls == 2
    assert sleeps == [30.0]


def test_http_429_honors_http_date_retry_after() -> None:
    """HTTP-date形式のRetry-Afterが最大30秒まで尊重されること。

    Arrange:
        - 初回に45秒後のHTTP-dateを返す429、2回目にvalid responseが用意される
    Act:
        - Scene Catalog推論が実行される
    Assert:
        - HTTP-dateの差分が解釈され待機が30秒へ制限されること
    """
    # Arrange
    calls = 0
    sleeps: list[float] = []

    def requester(
        _method: str,
        url: str,
        _payload: Mapping[str, object] | None,
        _timeout: float,
    ) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            headers = Message()
            retry_at = datetime.now(timezone.utc) + timedelta(seconds=45)
            headers["Retry-After"] = format_datetime(retry_at, usegmt=True)
            raise HTTPError(url, 429, "rate limited", headers, None)
        return _response(_catalog_payload())

    runtime = OllamaVisionRuntime(
        "http://localhost:11434",
        timeout_seconds=60.0,
        requester=requester,
        sleeper=sleeps.append,
        model_state_resolver=_resolved_artifact,
    )

    # Act
    catalog, diagnostics = runtime.create_scene_catalog(
        SceneCatalogRequest(
            representatives=(FrameCandidate("frame-a", b"image-a"),),
            selection_intent="ブログ画像を分類する",
        ),
        _resolved_model(ModelRole.SCENE_CATALOG),
        num_ctx=32768,
    )

    # Assert
    assert catalog.slugs[-1] == "other"
    assert diagnostics.attempt_count == 2
    assert calls == 2
    assert sleeps == [30.0]


def _catalog_payload() -> dict[str, object]:
    return {
        "scenes": [
            {
                "slug": "exploration",
                "display_name": "探索",
                "description": "フィールド探索",
                "selection_role": "ordinary",
            },
            {
                "slug": "battle",
                "display_name": "戦闘",
                "description": "繰り返される通常戦闘",
                "selection_role": "recurring_gameplay",
            },
            {
                "slug": "other",
                "display_name": "その他",
                "description": "分類不能",
                "selection_role": "ordinary",
            },
        ]
    }


def _annotation_payload() -> dict[str, object]:
    payload = _frame_observation_payload(
        (("frame-a", "battle", "event_dialogue", "high", "dialogue"),)
    )
    observation = _first_frame_observation(payload)
    observation["spoiler_risk"] = "high"
    observation["spoiler_evidence"] = "最終ボスの正体が画面で明示される"
    return payload


def _frame_observation_payload(
    rows: tuple[tuple[str, str, str, str, str], ...],
    *,
    context_relevance: str = "strong",
    supporting_context_cue_ids: tuple[str, ...] = ("cue-a",),
) -> dict[str, object]:
    return {
        "frame_observations": [
            {
                "frame_id": frame_id,
                "scene_slug": scene_slug,
                "content_kind": content_kind,
                "interface_kind": (
                    content_kind
                    if content_kind
                    in {
                        "shop",
                        "map",
                        "save",
                        "tutorial_help",
                        "other_interface",
                        "title",
                    }
                    else "none"
                ),
                "prominent_event_portrait": False,
                "visible_dialogue_text": content_kind == "event_dialogue",
                "visible_action": content_kind in {"gameplay_action", "event_action"},
                "visible_character_or_enemy": content_kind
                not in {"map", "save", "tutorial_help", "title"},
                "combat_action": scene_slug == "battle"
                and content_kind in {"gameplay_action", "event_action"},
                "visible_player_character": content_kind
                not in {"map", "save", "tutorial_help", "title"},
                "visible_opponent": scene_slug == "battle"
                and content_kind in {"gameplay_action", "event_action"},
                "explanation_value": explanation_value,
                "screen_text_kind": screen_text_kind,
                "primary_subject_visibility": "clear",
                "transient_obstruction": "none",
                "spoiler_risk": "none",
                "spoiler_evidence": "",
            }
            for (
                frame_id,
                scene_slug,
                content_kind,
                explanation_value,
                screen_text_kind,
            ) in rows
        ],
        "context_relevance": context_relevance,
        "supporting_context_cue_ids": list(supporting_context_cue_ids),
    }


def _response(content: dict[str, object]) -> dict[str, object]:
    return {
        "message": {"content": json.dumps(content, ensure_ascii=False)},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 123,
        "eval_count": 45,
    }


def _catalog() -> SceneCatalog:
    return SceneCatalog(
        (
            SceneCatalogEntry("exploration", "探索", "フィールド探索", "ordinary"),
            SceneCatalogEntry(
                "battle",
                "戦闘",
                "繰り返される通常戦闘",
                "recurring_gameplay",
            ),
            SceneCatalogEntry("other", "その他", "分類不能", "ordinary"),
        )
    )


def _catalog_with_battle_display(display_name: str) -> SceneCatalog:
    """battle表示名だけを置換したCatalogを返す。"""
    return SceneCatalog(
        tuple(
            SceneCatalogEntry(
                scene.slug,
                display_name if scene.slug == "battle" else scene.display_name,
                scene.description,
                scene.selection_role,
            )
            for scene in _catalog().scenes
        )
    )


def _annotation_request() -> CandidateAnnotationRequest:
    frame = FrameCandidate("frame-a", b"image-a")
    moment = CandidateMoment(
        identifier="mom_" + "a" * 64,
        source_pts=100,
        anchor_time=Fraction(10),
        timeline_segment_id="seg_" + "b" * 64,
        evidence=("scene",),
        proxy_quality_score=0.9,
        frame_candidate_ids=(frame.identifier,),
    )
    cue = ContextCue(
        identifier="cue-a",
        start=Fraction(9),
        end=Fraction(11),
        text="正体を明かす台詞",
    )
    return CandidateAnnotationRequest(
        moment=moment,
        frame_candidates=(frame,),
        context_cues=(cue,),
        video_set_progress=Fraction(1, 2),
        selection_intent="ブログ本文を説明できる画像を選ぶ",
        cue_selection_policy_version="nearby-context-v1",
    )


def _annotation_request_with_frame_ids(
    frame_ids: tuple[str, ...],
    *,
    include_context: bool = True,
) -> CandidateAnnotationRequest:
    frames = tuple(
        FrameCandidate(identifier, f"image-{identifier}".encode())
        for identifier in frame_ids
    )
    moment = CandidateMoment(
        identifier="mom_" + "a" * 64,
        source_pts=100,
        anchor_time=Fraction(10),
        timeline_segment_id="seg_" + "b" * 64,
        evidence=("scene",),
        proxy_quality_score=0.9,
        frame_candidate_ids=frame_ids,
    )
    cues = (
        (
            ContextCue(
                identifier="cue-a",
                start=Fraction(9),
                end=Fraction(11),
                text="正体を明かす台詞",
            ),
        )
        if include_context
        else ()
    )
    return CandidateAnnotationRequest(
        moment=moment,
        frame_candidates=frames,
        context_cues=cues,
        video_set_progress=Fraction(1, 2),
        selection_intent="ブログ本文を説明できる画像を選ぶ",
        cue_selection_policy_version="nearby-context-v1",
    )


def _annotation_request_with_context_text(text: str) -> CandidateAnnotationRequest:
    request = _annotation_request()
    cue = request.context_cues[0]
    return CandidateAnnotationRequest(
        moment=request.moment,
        frame_candidates=request.frame_candidates,
        context_cues=(
            ContextCue(
                identifier=cue.identifier,
                start=cue.start,
                end=cue.end,
                text=text,
            ),
        ),
        video_set_progress=request.video_set_progress,
        selection_intent=request.selection_intent,
        cue_selection_policy_version=request.cue_selection_policy_version,
    )


def _resolved_model(role: ModelRole) -> ResolvedModel:
    identity = ResolvedModelIdentity(ModelStoreKind.OLLAMA, "sha256:" + "a" * 64)
    return ResolvedModel(
        role=role,
        configured_name="qwen3-vl:8b-instruct",
        canonical_name="qwen3-vl:8b-instruct",
        local_identity_before_update=identity,
        update_status=ModelUpdateStatus.UNCHANGED,
        execution_identity=identity,
        runtime_identity=ModelRuntimeIdentity(ModelStoreKind.OLLAMA, "0.31.2"),
        artifact_location=None,
    )


def _resolved_artifact(model: ResolvedModel) -> ModelArtifact:
    return ModelArtifact(
        identity=model.execution_identity,
        canonical_name=model.canonical_name,
        runtime_identity=model.runtime_identity,
        location=None,
    )


def _first_message(payload: Mapping[str, object]) -> Mapping[str, object]:
    messages = payload.get("messages")
    assert isinstance(messages, list)
    assert messages
    message = messages[0]
    assert isinstance(message, dict)
    return cast(dict[str, object], message)


def _last_message(payload: Mapping[str, object]) -> Mapping[str, object]:
    messages = payload.get("messages")
    assert isinstance(messages, list)
    assert messages
    message = messages[-1]
    assert isinstance(message, dict)
    return cast(dict[str, object], message)


def _first_frame_observation(payload: Mapping[str, object]) -> dict[str, object]:
    observations = payload.get("frame_observations")
    assert isinstance(observations, list)
    assert observations
    observation = observations[0]
    assert isinstance(observation, dict)
    return cast(dict[str, object], observation)
