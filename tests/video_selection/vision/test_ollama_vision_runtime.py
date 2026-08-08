import base64
import io
import json
import time
from collections.abc import Mapping
from concurrent.futures import CancelledError
from datetime import datetime, timedelta, timezone
from email.message import Message
from email.utils import format_datetime
from fractions import Fraction
from threading import Event, Thread
from typing import cast
from urllib.error import HTTPError

import pytest
from PIL import Image

from src.video_selection.models.candidate_annotation import (
    candidate_annotation_free_text_is_safe,
)
from src.video_selection.models.candidate_annotation_request import (
    CandidateAnnotationRequest,
)
from src.video_selection.models.candidate_moment import CandidateMoment
from src.video_selection.models.combat_subject_evidence import CombatSubjectEvidence
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


def test_cancellation_stops_candidate_after_active_ollama_request() -> None:
    """active Ollama requestへ届いた中止要求でCandidate処理が停止されること。

    Arrange:
        - 応答前で停止するOllama requestとCandidate Annotationが用意される
    Act:
        - request開始後にCandidate Annotationの中止が要求される
    Assert:
        - 応答の解析やretryへ進まずCancelledErrorが返されること
    """
    # Arrange
    request_count = 0
    request_started = Event()
    release_request = Event()
    failures: list[BaseException] = []

    def requester(
        _method: str,
        _url: str,
        _payload: Mapping[str, object] | None,
        _timeout: float,
    ) -> object:
        nonlocal request_count
        request_count += 1
        request_started.set()
        if not release_request.wait(timeout=1.0):
            raise RuntimeError("Ollama requestを解放できませんでした")
        return {}

    runtime = OllamaVisionRuntime(
        "http://localhost:11434",
        timeout_seconds=60.0,
        requester=requester,
        sleeper=lambda _seconds: None,
        model_state_resolver=_resolved_artifact,
    )

    def annotate() -> None:
        try:
            runtime.annotate_candidate(
                _annotation_request(),
                _catalog(),
                _resolved_model(ModelRole.CANDIDATE_ANNOTATION),
                num_ctx=32768,
            )
        except BaseException as error:
            failures.append(error)

    worker = Thread(target=annotate, name="cancelled-candidate-annotation")

    # Act
    worker.start()
    assert request_started.wait(timeout=1.0)
    runtime.cancel_candidate_annotations()
    release_request.set()
    worker.join(timeout=1.0)

    # Assert
    assert worker.is_alive() is False
    assert len(failures) == 1
    assert isinstance(failures[0], CancelledError)
    assert "中止" in str(failures[0])
    assert request_count == 1


def test_cancellation_interrupts_candidate_retry_delay() -> None:
    """Candidate retry待機中の中止要求で待機が直ちに終了されること。

    Arrange:
        - 1秒のRetry-Afterを返すCandidate requestが用意される
    Act:
        - retry待機へ入った後にCandidate Annotationの中止が要求される
    Assert:
        - 待機時間を完了せずCancelledErrorで終了し再試行されないこと
    """
    # Arrange
    request_count = 0
    retry_response_returned = Event()
    failures: list[BaseException] = []

    def requester(
        _method: str,
        url: str,
        _payload: Mapping[str, object] | None,
        _timeout: float,
    ) -> object:
        nonlocal request_count
        request_count += 1
        headers = Message()
        headers["Retry-After"] = "1"
        retry_response_returned.set()
        raise HTTPError(url, 429, "rate limited", headers, None)

    runtime = OllamaVisionRuntime(
        "http://localhost:11434",
        timeout_seconds=60.0,
        requester=requester,
        model_state_resolver=_resolved_artifact,
    )

    def annotate() -> None:
        try:
            runtime.annotate_candidate(
                _annotation_request(),
                _catalog(),
                _resolved_model(ModelRole.CANDIDATE_ANNOTATION),
                num_ctx=32768,
            )
        except BaseException as error:
            failures.append(error)

    worker = Thread(target=annotate, name="cancelled-candidate-retry")

    # Act
    worker.start()
    assert retry_response_returned.wait(timeout=1.0)
    time.sleep(0.1)
    runtime.cancel_candidate_annotations()
    worker.join(timeout=0.2)

    # Assert
    assert worker.is_alive() is False
    assert len(failures) == 1
    assert isinstance(failures[0], CancelledError)
    assert request_count == 1


def test_scene_catalog_uses_strict_documented_ollama_request() -> None:
    """Scene Catalogが画像付きstrict schema requestから生成されること。

    Arrange:
        - 3件のsceneを返すfake Ollama APIと2枚の代表画像が用意される
    Act:
        - Scene Catalog推論が実行される
    Assert:
        - JSON Schema object、temperature 0、seed 0、stream/think falseが送られること
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
    assert payload["options"] == {"temperature": 0, "num_ctx": 32768, "seed": 0}
    schema = payload["format"]
    assert isinstance(schema, dict)
    assert schema["additionalProperties"] is False
    scene_properties = schema["properties"]["scenes"]["items"]["properties"]
    assert scene_properties["scene_kind"]["enum"] == [
        "combat",
        "exploration",
        "interface",
        "event",
        "other",
    ]
    assert "quality_score" not in json.dumps(schema)
    assert "final_score" not in json.dumps(schema)
    assert "selected" not in json.dumps(schema)
    messages = payload["messages"]
    assert isinstance(messages, list)
    assert messages[0]["images"] == ["aW1hZ2UtYQ==", "aW1hZ2UtYg=="]
    prompt = messages[0]["content"]
    assert isinstance(prompt, str)
    assert "scene_kindはcombat=" in prompt
    assert "scene_kindは複数sceneで重複して構いません" in prompt
    assert "slugはcatalog内で一意にします" in prompt
    assert "recurring_gameplay=" in prompt
    assert "同じ画面構造を一時的な敵やエフェクトだけで別sceneへ分割しません" in prompt
    assert "どの画像にも当てはまる再利用可能なカテゴリ表現" in prompt
    assert "町・ダンジョンなどの場所" in prompt


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


def test_duplicate_scene_slug_is_repaired_with_explicit_contract() -> None:
    """重複Scene Slugが一意性の修復指示付きで再試行されること。

    Arrange:
        - 初回だけScene Kindをslugとして重複させたCatalog応答が用意される
    Act:
        - Scene Catalog推論が実行される
    Assert:
        - Scene Kindの重複を許しつつslugを一意にする指示で修復されること
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
        response = _catalog_payload()
        if len(payloads) == 1:
            scenes = response["scenes"]
            assert isinstance(scenes, list)
            second_scene = scenes[1]
            assert isinstance(second_scene, dict)
            second_scene["slug"] = "exploration"
        return _response(response)

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
    assert catalog.slugs == ("exploration", "battle", "other")
    assert diagnostics.attempt_count == 2
    assert diagnostics.validation_code == "scene_catalog_domain_invalid"
    second_prompt = _first_message(payloads[1])["content"]
    assert isinstance(second_prompt, str)
    assert "scene_kindは重複可能" in second_prompt
    assert "一意なslug" in second_prompt


def test_duplicate_scene_slug_is_deterministically_suffixed_after_retry() -> None:
    """再試行でも重複する非other slugが決定的に一意化されること。

    Arrange:
        - 二回とも異なるScene Kindへ同じslugを返すCatalog応答が用意される
    Act:
        - Scene Catalog推論が実行される
    Assert:
        - 二回目の重複slugへ入力順のsuffixが付けられること
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
        response = _catalog_payload()
        scenes = response["scenes"]
        assert isinstance(scenes, list)
        second_scene = scenes[1]
        assert isinstance(second_scene, dict)
        second_scene["slug"] = "exploration"
        return _response(response)

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
    assert catalog.slugs == ("exploration", "exploration-2", "other")
    assert diagnostics.attempt_count == 2
    assert diagnostics.validation_code == "scene_catalog_domain_invalid"
    assert len(payloads) == 2


def test_other_scene_slug_is_canonicalized_from_scene_kind() -> None:
    """Scene Kind otherの自由なslugが分類の逃げ先へ正規化されること。

    Arrange:
        - other sceneへ具体化された自由なslugを返すCatalog応答が用意される
    Act:
        - Scene Catalog推論が実行される
    Assert:
        - Scene Kindを根拠にslugが正確なotherへ正規化されること
    """

    # Arrange
    def requester(
        _method: str,
        _url: str,
        _payload: Mapping[str, object] | None,
        _timeout: float,
    ) -> object:
        response = _catalog_payload()
        scenes = response["scenes"]
        assert isinstance(scenes, list)
        other_scene = scenes[2]
        assert isinstance(other_scene, dict)
        other_scene["slug"] = "other-ordinary-scene"
        other_scene["display_name"] = "通常の探索シーン"
        other_scene["description"] = "石畳の町で探索する場面"
        return _response(response)

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
    assert catalog.slugs == ("exploration", "battle", "other")
    other = catalog.for_slug("other")
    assert other.display_name == "その他"
    assert other.description == "共有Scene Catalogの他sceneに分類できない場面"
    assert other.scene_kind == "other"
    assert diagnostics.attempt_count == 1


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
    assert observation_properties["scene_catalog_match"]["type"] == "boolean"
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
    assert observation_properties["cinematic_event_presentation"] == {"type": "boolean"}
    assert "visible_dialogue_text" not in observation_properties
    assert observation_properties["on_screen_dialogue_text_visible"]["type"] == (
        "boolean"
    )
    assert observation_properties["dialogue_text_presentation"]["enum"] == [
        "none",
        "dialogue_box",
        "speech_bubble",
        "subtitle_overlay",
        "other",
    ]
    assert observation_properties["visible_action"] == {"type": "boolean"}
    assert observation_properties["visible_character_or_enemy"] == {"type": "boolean"}
    assert observation_properties["combat_encounter_kind"]["enum"] == [
        "not_combat",
        "ordinary",
        "major",
        "uncertain",
    ]
    assert observation_properties["combat_encounter_basis"]["enum"] == [
        "none",
        "ordinary_opponent_presentation",
        "ordinary_encounter_presentation",
        "major_opponent_presentation",
        "major_encounter_presentation",
        "ambiguous",
    ]
    subject_evidence = observation_properties["combat_subject_evidence"]
    assert subject_evidence["additionalProperties"] is False
    assert subject_evidence["properties"]["body_plan"]["enum"] == [
        "unknown",
        "humanoid",
        "quadruped",
        "serpentine",
        "fish_like",
        "insectoid",
        "avian",
        "amorphous",
        "plant_like",
        "mechanical",
        "structure",
        "multi_part",
        "other",
    ]
    assert subject_evidence["properties"]["colors"]["maxItems"] == 2
    assert subject_evidence["properties"]["traits"]["maxItems"] == 4
    assert observation_properties["player_body_visibility"]["enum"] == [
        "clear",
        "partial",
        "absent",
    ]
    assert observation_properties["opponent_body_visibility"]["enum"] == [
        "clear",
        "partial",
        "absent",
    ]
    assert observation_properties["effect_only_frame"]["type"] == "boolean"
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
    assert "on_screen_dialogue_text_visible" in prompt
    assert "dialogue_text_presentation" in prompt
    assert "visible_action" in prompt
    assert "visible_character_or_enemy" in prompt
    assert "combat_encounter_kind" in prompt
    assert "combat_encounter_basis" in prompt
    assert "積極的な画像内根拠がある場合だけordinary" in prompt
    assert "敵名やHP・status barだけではordinaryにもmajorにも" in prompt
    assert "どちらの積極的根拠もない場合はuncertain" in prompt
    assert "spoiler_riskとは独立" in prompt
    assert "player_body_visibility" in prompt
    assert "opponent_body_visibility" in prompt
    assert "effect_only_frame" in prompt
    assert "人物portraitだけ" in prompt
    assert "手紙・手記・日誌・記録" in prompt
    assert "prominent_event_portrait" in prompt
    assert "cinematic_event_presentation" in prompt
    assert "上下の映画的な黒帯" in prompt
    assert "画面隅の小さな" in prompt
    assert "scene_catalog_match" in prompt
    assert "音声、Context Cue、Video Set Progressを一致の根拠にしません" in prompt
    assert "explanation_valueのnone=" in prompt
    assert "context_relevanceのstrong=" in prompt
    assert "spoiler_riskのhigh=" in prompt
    messages = payloads[0]["messages"]
    assert isinstance(messages, list)
    frame_prompt = messages[0]["content"]
    assert "combat_subject_evidence" in frame_prompt
    assert "固有名" in frame_prompt
    assert "Context Cue、前後frameを根拠に" in frame_prompt
    assert "distinctiveness=distinctive" in frame_prompt


def test_candidate_annotation_keeps_single_image_combat_subject_evidence() -> None:
    """一枚の画像から得た戦闘対象の外見根拠がAnnotationへ保持されること。

    Arrange:
        - 説明価値なしの主要戦闘frameと構造化された外見観測が用意される
    Act:
        - 公開Vision RuntimeでCandidate Annotationが実行される
    Assert:
        - 名前を含まない外見根拠がCandidate Annotationへ保持されること
        - requestには一枚の画像だけが渡されること
    """
    # Arrange
    payloads: list[Mapping[str, object]] = []
    response = _frame_observation_payload(
        (("frame-a", "battle", "gameplay_action", "none", "hud"),)
    )
    observation = _first_frame_observation(response)
    observation["combat_encounter_kind"] = "major"
    observation["combat_encounter_basis"] = "major_opponent_presentation"
    observation["combat_subject_evidence"] = {
        "body_plan": "quadruped",
        "scale": "large",
        "surface": "organic",
        "colors": ["green", "brown"],
        "traits": ["large_mouth", "bulbous_body"],
        "distinctiveness": "distinctive",
    }

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
    annotation, _diagnostics = runtime.annotate_candidate(
        _annotation_request(),
        _catalog(),
        _resolved_model(ModelRole.CANDIDATE_ANNOTATION),
        num_ctx=32768,
    )

    # Assert
    assert annotation.combat_subject_evidence == CombatSubjectEvidence(
        body_plan="quadruped",
        scale="large",
        surface="organic",
        colors=("brown", "green"),
        traits=("bulbous_body", "large_mouth"),
        distinctiveness="distinctive",
    )
    messages = payloads[0]["messages"]
    assert isinstance(messages, list)
    assert sum("images" in message for message in messages) == 1


def test_incomplete_distinctive_combat_subject_evidence_is_retried() -> None:
    """不完全なdistinctive Combat Subject Evidenceがdomain retryされること。

    Arrange:
        - 初回は有限enumだが色と特徴が空のdistinctive応答が用意される
        - 2回目は完全なdistinctive応答が用意される
    Act:
        - 公開Vision RuntimeでCandidate Annotationが実行される
    Assert:
        - 初回のdomain違反で処理全体が中断されず再試行されること
        - 2回目の完全なEvidenceとstable validation codeが返されること
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
        response = _frame_observation_payload(
            (("frame-a", "exploration", "gameplay_idle", "high", "hud"),)
        )
        observation = _first_frame_observation(response)
        observation["combat_subject_evidence"] = {
            "body_plan": "quadruped",
            "scale": "large",
            "surface": "organic",
            "colors": [] if len(payloads) == 1 else ["green"],
            "traits": [] if len(payloads) == 1 else ["large_mouth"],
            "distinctiveness": "distinctive",
        }
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
        _annotation_request(),
        _catalog(),
        _resolved_model(ModelRole.CANDIDATE_ANNOTATION),
        num_ctx=32768,
    )

    # Assert
    assert annotation.combat_subject_evidence == CombatSubjectEvidence(
        body_plan="quadruped",
        scale="large",
        surface="organic",
        colors=("green",),
        traits=("large_mouth",),
        distinctiveness="distinctive",
    )
    assert diagnostics.attempt_count == 2
    assert diagnostics.validation_code == "candidate_annotation_domain_invalid"
    assert len(payloads) == 2
    second_prompt = _last_message(payloads[1])["content"]
    assert isinstance(second_prompt, str)
    assert "candidate_annotation_domain_invalid" in second_prompt


def test_candidate_uses_generic_scene_when_catalog_details_do_not_match_frame() -> None:
    """Catalogの具体的な場所が画像で確認できなければ要約から除かれること。

    Arrange:
        - 町でのevent sceneと、ダンジョンに見える会話frameの観測が用意される
        - 画像がScene Catalogの具体的な説明と一致しないことが返される
    Act:
        - Candidate Annotation推論が実行される
    Assert:
        - Sceneが分類の逃げ先へ正規化されること
        - 町と断定しない画像単体の要約が返されること
    """
    # Arrange
    catalog = SceneCatalog(
        (
            SceneCatalogEntry(
                "exploration",
                "探索",
                "フィールド探索",
                "exploration",
                "ordinary",
            ),
            SceneCatalogEntry(
                "event-town",
                "町での会話イベント",
                "町の広場で人物が会話するevent",
                "event",
                "ordinary",
            ),
            SceneCatalogEntry("other", "その他", "分類不能", "other", "ordinary"),
        )
    )
    response = _frame_observation_payload(
        (("frame-a", "event-town", "event_dialogue", "high", "dialogue"),)
    )
    _first_frame_observation(response)["scene_catalog_match"] = False
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
        catalog,
        _resolved_model(ModelRole.CANDIDATE_ANNOTATION),
        num_ctx=32768,
    )

    # Assert
    assert annotation.scene_slug == "other"
    assert annotation.summary == "画面内テキストのあるイベント"


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
            "on_screen_dialogue_text_visible": True,
            "dialogue_text_presentation": "dialogue_box",
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
        "cinematic_event_presentation",
        "expected_blog_image_type",
    ),
    (
        ("other", "document", False, False, "menu"),
        ("gameplay_idle", "none", True, False, "event"),
        ("gameplay_idle", "none", False, True, "event"),
    ),
)
def test_static_document_and_silent_event_presentation_are_not_eligible(
    content_kind: str,
    interface_kind: str,
    prominent_event_portrait: bool,
    cinematic_event_presentation: bool,
    expected_blog_image_type: str,
) -> None:
    """静止文書と台詞のないイベント演出が掲載不可にされること。

    Arrange:
        - 高評価だが静止文書または台詞のないイベント演出の応答が用意される
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
            "cinematic_event_presentation": cinematic_event_presentation,
            "on_screen_dialogue_text_visible": False,
            "dialogue_text_presentation": "none",
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


def test_combat_without_visible_opponent_has_no_explanation_value() -> None:
    """敵本体が見えないframeがmodelの高評価より優先して除外されること。

    Arrange:
        - 戦闘中だが発光で敵本体が判別できないhigh応答が用意される
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
            "combat_encounter_kind": "ordinary",
            "player_body_visibility": "clear",
            "opponent_body_visibility": "absent",
            "effect_only_frame": False,
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


@pytest.mark.parametrize(
    ("confirmed_visible", "expected_explanation_value"),
    ((False, "none"), (True, "high")),
)
def test_contextual_cinematic_dialogue_is_visually_rechecked(
    confirmed_visible: bool,
    expected_explanation_value: str,
) -> None:
    """文脈付きイベントの画面内台詞が画像に対して再確認されること。

    Arrange:
        - 初回は音声会話を画面内台詞としたイベント応答が用意される
        - 再確認では画像上の台詞文字がある場合とない場合が用意される
    Act:
        - Candidate Annotation推論が実行される
    Assert:
        - 視覚再確認後の台詞表示だけで説明価値が決定されること
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
        response = _frame_observation_payload(
            (("frame-a", "exploration", "event_dialogue", "high", "dialogue"),)
        )
        observation = _first_frame_observation(response)
        visible = True if len(payloads) == 1 else confirmed_visible
        observation.update(
            {
                "cinematic_event_presentation": True,
                "on_screen_dialogue_text_visible": visible,
                "dialogue_text_presentation": ("dialogue_box" if visible else "none"),
            }
        )
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
        _annotation_request(),
        _catalog(),
        _resolved_model(ModelRole.CANDIDATE_ANNOTATION),
        num_ctx=32768,
    )

    # Assert
    assert annotation.explanation_value == expected_explanation_value
    assert diagnostics.attempt_count == 2
    assert diagnostics.validation_code == (
        "candidate_annotation_dialogue_visibility_unverified"
    )
    second_prompt = _last_message(payloads[1])["content"]
    assert isinstance(second_prompt, str)
    assert "candidate_annotation_dialogue_visibility_unverified" in second_prompt
    assert "音声やContext Cueを根拠にしません" in second_prompt
    assert "spoiler_riskがnoneならspoiler_evidenceは空文字列" in second_prompt
    assert "weakまたはstrongなら入力内IDを1件以上" in second_prompt


def test_invalid_candidate_response_does_not_count_as_dialogue_verification() -> None:
    """parse不能な応答が画面内台詞の独立確認として数えられないこと。

    Arrange:
        - 未知frameを返す初回応答と画面内台詞を初めて主張する2回目が用意される
        - 3回目では画像上に台詞文字がないと訂正される
    Act:
        - Candidate Annotation推論が実行される
    Assert:
        - parse成功した応答だけが数えられ3回目の独立確認結果が採用されること
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
        response = _frame_observation_payload(
            (("frame-a", "exploration", "event_dialogue", "high", "dialogue"),)
        )
        observation = _first_frame_observation(response)
        if len(payloads) == 1:
            observation["frame_id"] = "foreign-frame"
        else:
            visible = len(payloads) == 2
            observation.update(
                {
                    "cinematic_event_presentation": True,
                    "on_screen_dialogue_text_visible": visible,
                    "dialogue_text_presentation": (
                        "dialogue_box" if visible else "none"
                    ),
                }
            )
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
        _annotation_request(),
        _catalog(),
        _resolved_model(ModelRole.CANDIDATE_ANNOTATION),
        num_ctx=32768,
    )

    # Assert
    assert annotation.explanation_value == "none"
    assert diagnostics.attempt_count == 3
    assert diagnostics.validation_code == (
        "candidate_annotation_dialogue_visibility_unverified"
    )
    assert len(payloads) == 3


@pytest.mark.parametrize(
    (
        "confirmed_opponent_visibility",
        "confirmed_opponent_framing",
        "confirmed_opponent_presentation",
        "confirmed_combat_interaction_visibility",
        "confirmed_effect_only",
        "expected_value",
        "expected_attempt_count",
    ),
    (
        ("clear", "complete", "prominent", "none", False, "high", 5),
        ("clear", "complete", "recognizable", "direct", False, "high", 5),
        ("clear", "complete", "recognizable", "indirect", False, "none", 3),
        ("clear", "complete", "weak", "direct", False, "none", 3),
        ("clear", "edge_cropped", "prominent", "direct", False, "none", 3),
        ("absent", "absent", "absent", "none", False, "none", 3),
        ("clear", "complete", "prominent", "direct", True, "none", 3),
    ),
)
def test_publishable_combat_visibility_is_visually_rechecked(
    confirmed_opponent_visibility: str,
    confirmed_opponent_framing: str,
    confirmed_opponent_presentation: str,
    confirmed_combat_interaction_visibility: str,
    confirmed_effect_only: bool,
    expected_value: str,
    expected_attempt_count: int,
) -> None:
    """掲載可能とされた戦闘の敵本体、構図、エフェクトが再確認されること。

    Arrange:
        - 初回は敵本体が明瞭でエフェクトだけではないとする戦闘応答が用意される
        - 再確認では敵本体、画面端の欠け、エフェクトだけという観測が変更される
    Act:
        - Candidate Annotation推論が実行される
    Assert:
        - 再確認後の直接観測だけでExplanation Valueが決定されること
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
            response = _frame_observation_payload(
                (("frame-a", "battle", "gameplay_action", "high", "hud"),)
            )
            observation = _first_frame_observation(response)
            observation["spoiler_risk"] = "medium"
            observation["spoiler_evidence"] = "物語上の進行情報が画面に示される"
            return _response(response)
        if len(payloads) == 2:
            return _response(_combat_encounter_payload(visible=True, evidence="both"))
        if len(payloads) == 5:
            return _response(_combat_visibility_edge_audit_payload())
        return _response(
            {
                "effect_screen_coverage": "over_half",
                "largest_foreground_element": "visual_effect",
                "player_body_visibility": "partial",
                "opponent_body_visibility": confirmed_opponent_visibility,
                "opponent_body_framing": confirmed_opponent_framing,
                "opponent_presentation": confirmed_opponent_presentation,
                "combat_interaction_visibility": (
                    confirmed_combat_interaction_visibility
                ),
                "effect_overlaps_combatant_body": "severe",
                "effect_only_frame": confirmed_effect_only,
            }
        )

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
    assert annotation.explanation_value == expected_value
    assert annotation.combat_action is True
    assert annotation.combat_encounter_kind == (
        "ordinary" if expected_value == "high" else "uncertain"
    )
    assert annotation.combat_encounter_basis == (
        "ordinary_opponent_presentation" if expected_value == "high" else "ambiguous"
    )
    assert annotation.summary == "戦闘の具体的なプレイ"
    assert annotation.spoiler_risk == "medium"
    assert diagnostics.attempt_count == expected_attempt_count
    assert diagnostics.validation_code is None
    verification_schema = payloads[2]["format"]
    assert isinstance(verification_schema, Mapping)
    verification_properties = verification_schema.get("properties")
    assert isinstance(verification_properties, Mapping)
    assert set(verification_properties) == {
        "effect_screen_coverage",
        "largest_foreground_element",
        "player_body_visibility",
        "opponent_body_visibility",
        "opponent_body_framing",
        "opponent_presentation",
        "combat_interaction_visibility",
        "effect_overlaps_combatant_body",
        "effect_only_frame",
    }
    second_prompt = _last_message(payloads[2])["content"]
    assert isinstance(second_prompt, str)
    assert "この画像1枚に実際に見える画素だけ" in second_prompt
    assert "輪郭と姿勢" in second_prompt
    assert "画像の端で大きく切れる" in second_prompt
    assert "opponent_body_framing" in second_prompt
    assert "音声、前後場面、説明文は使いません" in second_prompt
    assert "見下ろし型・遠景・非人型" in second_prompt
    assert "最も明瞭に判別でき、最も完全に収まる一体" in second_prompt
    assert "別の敵が画面端にいるだけ" in second_prompt
    assert "攻撃エフェクトがplayer付近にあるだけ" in second_prompt
    assert "opponent_presentation" in second_prompt
    assert "combat_interaction_visibility" in second_prompt
    if expected_attempt_count == 5:
        confirmation_prompt = _last_message(payloads[3])["content"]
        assert isinstance(confirmation_prompt, str)
        assert "掲載可否を確定する独立した再確認" in confirmation_prompt
        assert "先の回答を推測せず" in confirmation_prompt
        edge_audit_prompt = _last_message(payloads[4])["content"]
        assert isinstance(edge_audit_prompt, str)
        assert "外周30%を切り出した診断画像" in edge_audit_prompt
        assert "順番はtop、bottom、left、right" in edge_audit_prompt


def test_clean_combat_visibility_is_confirmed_before_publication() -> None:
    """可視性を確認できない戦闘の主種別がartifactへ残されないこと。

    Arrange:
        - 主推論がordinary、戦闘有無専用確認がmajorとした応答が用意される
        - 独立確認では敵本体が画面端で欠ける応答が用意される
    Act:
        - Candidate Annotation推論が実行される
    Assert:
        - 掲載不可となり未検証のordinaryも専用確認のmajorも保存されないこと
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
            return _response(
                _frame_observation_payload(
                    (("frame-a", "battle", "gameplay_action", "high", "hud"),)
                )
            )
        if len(payloads) == 2:
            return _response(
                _combat_encounter_payload(
                    visible=True,
                    evidence="both",
                    combat_encounter_kind="major",
                    combat_encounter_basis="major_opponent_presentation",
                )
            )
        verification = {
            "effect_screen_coverage": "under_quarter",
            "largest_foreground_element": "opponent_body",
            "player_body_visibility": "clear",
            "opponent_body_visibility": "clear",
            "opponent_body_framing": "complete",
            "opponent_presentation": "prominent",
            "combat_interaction_visibility": "direct",
            "effect_overlaps_combatant_body": "partial",
            "effect_only_frame": False,
        }
        if len(payloads) == 4:
            verification["largest_foreground_element"] = "visual_effect"
            verification["opponent_body_visibility"] = "partial"
            verification["opponent_body_framing"] = "edge_cropped"
            verification["effect_overlaps_combatant_body"] = "severe"
        return _response(verification)

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
    assert annotation.explanation_value == "none"
    assert annotation.combat_action is True
    assert annotation.combat_encounter_kind == "uncertain"
    assert annotation.combat_encounter_basis == "ambiguous"
    assert annotation.selection_coverage_facet is None
    assert diagnostics.attempt_count == 4
    confirmation_prompt = _last_message(payloads[3])["content"]
    assert isinstance(confirmation_prompt, str)
    assert "画像の画素を最初から観測し直してください" in confirmation_prompt


def test_combat_edge_audit_rejects_cropped_opponent_after_false_positives() -> None:
    """二回の可視性誤判定後も外周strip監査で欠けた敵が掲載不可にされること。

    Arrange:
        - 主推論と二回の可視性確認で掲載可能と誤判定される戦闘が用意される
        - 外周strip監査では敵本体が実際の外端へ到達する応答が用意される
    Act:
        - Candidate Annotation推論が実行される
    Assert:
        - 外周stripの直接観測によりExplanation Valueがnoneにされること
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
            return _response(
                _frame_observation_payload(
                    (("frame-a", "battle", "gameplay_action", "high", "hud"),)
                )
            )
        if len(payloads) == 2:
            return _response(_combat_encounter_payload(visible=True, evidence="both"))
        if len(payloads) == 5:
            return _response(_combat_visibility_edge_audit_payload(("right",)))
        return _response(
            _combat_visibility_payload(
                opponent_body_visibility="clear",
                opponent_body_framing="complete",
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
    annotation, diagnostics = runtime.annotate_candidate(
        _annotation_request(),
        _catalog(),
        _resolved_model(ModelRole.CANDIDATE_ANNOTATION),
        num_ctx=32768,
    )

    # Assert
    assert annotation.explanation_value == "none"
    assert diagnostics.attempt_count == 5
    edge_audit_prompt = _last_message(payloads[4])["content"]
    assert isinstance(edge_audit_prompt, str)
    assert "最初の画像は元スクリーンショット全体" in edge_audit_prompt
    assert "最も明瞭に判別でき、最も完全に収まる一体" in edge_audit_prompt
    assert "別の攻撃相手が外端へ到達しても数えません" in edge_audit_prompt
    assert "残り4画像は同じ画像から外周30%を切り出した診断画像" in edge_audit_prompt
    assert "診断用の内側crop境界に触れることは数えません" in edge_audit_prompt
    edge_images = _last_message(payloads[4])["images"]
    assert isinstance(edge_images, list)
    assert all(isinstance(item, str) for item in edge_images)
    image_sizes = []
    for encoded_image in edge_images:
        assert isinstance(encoded_image, str)
        with Image.open(io.BytesIO(base64.b64decode(encoded_image))) as image:
            image_sizes.append(image.size)
    assert image_sizes == [(20, 10), (20, 3), (20, 3), (6, 10), (6, 10)]


def test_combat_edge_audit_schema_failure_is_retried() -> None:
    """外周strip監査の不正な辺一覧が一回だけ再試行されること。

    Arrange:
        - 二回の可視性確認で掲載可能とされる戦闘が用意される
        - 初回の外周strip監査では辺が欠け、再試行では4辺が返される
    Act:
        - Candidate Annotation推論が実行される
    Assert:
        - stable validation code付きで修復され掲載価値が保持されること
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
            return _response(
                _frame_observation_payload(
                    (("frame-a", "battle", "gameplay_action", "high", "hud"),)
                )
            )
        if len(payloads) == 2:
            return _response(_combat_encounter_payload(visible=True, evidence="both"))
        if len(payloads) == 5:
            return _response({"edges": []})
        if len(payloads) == 6:
            return _response(_combat_visibility_edge_audit_payload())
        return _response(
            _combat_visibility_payload(
                opponent_body_visibility="clear",
                opponent_body_framing="complete",
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
    annotation, diagnostics = runtime.annotate_candidate(
        _annotation_request(),
        _catalog(),
        _resolved_model(ModelRole.CANDIDATE_ANNOTATION),
        num_ctx=32768,
    )

    # Assert
    assert annotation.explanation_value == "high"
    assert diagnostics.attempt_count == 6
    assert diagnostics.validation_code == "combat_visibility_edge_audit_schema_invalid"
    repair_prompt = _last_message(payloads[5])["content"]
    assert isinstance(repair_prompt, str)
    assert "combat_visibility_edge_audit_schema_invalid" in repair_prompt


def test_possible_combat_is_routed_before_visibility_verification() -> None:
    """主推論が戦闘flagを落としたboss戦も敵可視性が確認されること。

    Arrange:
        - recurring gameplayのactionを非戦闘と誤分類した掲載可能応答が用意される
        - 敵status UIで戦闘と確認され、敵本体なしとする可視性応答が用意される
    Act:
        - Candidate Annotation推論が実行される
    Assert:
        - 戦闘有無と敵可視性が別々に確認され、掲載・通常戦闘対象外にされること
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
            response = _frame_observation_payload(
                (("frame-a", "exploration", "gameplay_action", "high", "hud"),)
            )
            observation = _first_frame_observation(response)
            observation["combat_encounter_kind"] = "not_combat"
            observation["combat_encounter_basis"] = "none"
            observation["opponent_body_visibility"] = "absent"
            return _response(response)
        if len(payloads) == 2:
            return _response(
                _combat_encounter_payload(
                    visible=True,
                    evidence="enemy_status_ui",
                )
            )
        return _response(
            {
                "effect_screen_coverage": "quarter_to_half",
                "largest_foreground_element": "visual_effect",
                "player_body_visibility": "partial",
                "opponent_body_visibility": "absent",
                "opponent_body_framing": "absent",
                "opponent_presentation": "absent",
                "combat_interaction_visibility": "none",
                "effect_overlaps_combatant_body": "severe",
                "effect_only_frame": False,
            }
        )

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
        _catalog_with_recurring_exploration(),
        _resolved_model(ModelRole.CANDIDATE_ANNOTATION),
        num_ctx=32768,
    )

    # Assert
    assert annotation.explanation_value == "none"
    assert annotation.combat_action is False
    assert diagnostics.attempt_count == 3
    assert diagnostics.validation_code is None
    routing_schema = payloads[1]["format"]
    assert isinstance(routing_schema, Mapping)
    routing_properties = routing_schema.get("properties")
    assert isinstance(routing_properties, Mapping)
    assert set(routing_properties) == {
        "combat_encounter_kind",
        "combat_encounter_basis",
        "combat_encounter_evidence",
    }
    routing_prompt = _last_message(payloads[1])["content"]
    assert isinstance(routing_prompt, str)
    assert "敵またはboss固有の名前とHP・status bar" in routing_prompt
    assert "敵本体が画面端で切れる" in routing_prompt
    visibility_prompt = _last_message(payloads[2])["content"]
    assert isinstance(visibility_prompt, str)
    assert "opponent_body_framing" in visibility_prompt


def test_enemy_status_without_visible_player_does_not_mark_combat_action() -> None:
    """敵status UIだけの場面が通常戦闘として注釈されないこと。

    Arrange:
        - 主推論が非戦闘としたrecurring gameplay actionが用意される
        - 敵status UIで戦闘と判定されるがplayer本体不在の可視性応答が用意される
    Act:
        - Candidate Annotation推論が実行される
    Assert:
        - playerと敵の戦闘を直接確認できず掲載価値とcombat actionが無効になること
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
            response = _frame_observation_payload(
                (("frame-a", "exploration", "gameplay_action", "high", "hud"),)
            )
            observation = _first_frame_observation(response)
            observation["combat_encounter_kind"] = "not_combat"
            observation["combat_encounter_basis"] = "none"
            observation["opponent_body_visibility"] = "absent"
            return _response(response)
        if len(payloads) == 2:
            return _response(
                _combat_encounter_payload(
                    visible=True,
                    evidence="enemy_status_ui",
                )
            )
        if len(payloads) == 5:
            return _response(_combat_visibility_edge_audit_payload())
        return _response(
            {
                "effect_screen_coverage": "under_quarter",
                "largest_foreground_element": "opponent_body",
                "player_body_visibility": "absent",
                "opponent_body_visibility": "clear",
                "opponent_body_framing": "complete",
                "opponent_presentation": "prominent",
                "combat_interaction_visibility": "indirect",
                "effect_overlaps_combatant_body": "none",
                "effect_only_frame": False,
            }
        )

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
        _catalog_with_recurring_exploration(),
        _resolved_model(ModelRole.CANDIDATE_ANNOTATION),
        num_ctx=32768,
    )

    # Assert
    assert annotation.explanation_value == "none"
    assert annotation.combat_action is False
    assert diagnostics.attempt_count == 3
    assert len(payloads) == 3


def test_noncombat_recurring_action_is_cross_checked_by_visibility() -> None:
    """非戦闘のrecurring actionが敵可視性でも再確認されること。

    Arrange:
        - recurring gameplayの非戦闘action応答が用意される
        - 戦闘なしと敵本体なしが各二回返される専用応答が用意される
    Act:
        - Candidate Annotation推論が実行される
    Assert:
        - 両方の観測で戦闘が否定された場合だけ掲載価値が保持されること
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
            return _response(
                _frame_observation_payload(
                    (("frame-a", "exploration", "gameplay_action", "high", "hud"),)
                )
            )
        if len(payloads) <= 3:
            return _response(_combat_encounter_payload(visible=False, evidence="none"))
        return _response(_combat_visibility_payload())

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
        _catalog_with_recurring_exploration(),
        _resolved_model(ModelRole.CANDIDATE_ANNOTATION),
        num_ctx=32768,
    )

    # Assert
    assert annotation.explanation_value == "high"
    assert diagnostics.attempt_count == 5
    assert diagnostics.validation_code is None
    assert len(payloads) == 5
    confirmation_prompt = _last_message(payloads[2])["content"]
    assert isinstance(confirmation_prompt, str)
    assert "掲載可否を確定する独立した再確認" in confirmation_prompt
    visibility_prompt = _last_message(payloads[3])["content"]
    assert isinstance(visibility_prompt, str)
    assert "opponent_body_framing" in visibility_prompt
    visibility_confirmation_prompt = _last_message(payloads[4])["content"]
    assert isinstance(visibility_confirmation_prompt, str)
    assert "掲載可否を確定する独立した再確認" in visibility_confirmation_prompt


def test_status_only_combat_stays_uncertain_without_positive_basis() -> None:
    """敵status UIだけの戦闘が通常戦闘coverageへ入らないこと。

    Arrange:
        - 主推論が非戦闘としたcombat sceneのframeが用意される
        - 専用確認では敵status UIから戦闘を確認できるが種別根拠は曖昧とされる
        - playerと敵本体の掲載可能な可視性が二回確認される
    Act:
        - Candidate Annotation推論が実行される
    Assert:
        - 戦闘はuncertainとして保持されordinary_combat facetを持たないこと
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
            response = _frame_observation_payload(
                (("frame-a", "battle", "gameplay_action", "high", "hud"),)
            )
            observation = _first_frame_observation(response)
            observation["combat_encounter_kind"] = "not_combat"
            observation["combat_encounter_basis"] = "none"
            observation["opponent_body_visibility"] = "absent"
            return _response(response)
        if len(payloads) == 2:
            return _response(
                _combat_encounter_payload(
                    visible=True,
                    evidence="enemy_status_ui",
                    combat_encounter_kind="uncertain",
                    combat_encounter_basis="ambiguous",
                )
            )
        if len(payloads) == 5:
            return _response(_combat_visibility_edge_audit_payload())
        return _response(
            {
                "effect_screen_coverage": "under_quarter",
                "largest_foreground_element": "opponent_body",
                "player_body_visibility": "clear",
                "opponent_body_visibility": "clear",
                "opponent_body_framing": "complete",
                "opponent_presentation": "prominent",
                "combat_interaction_visibility": "direct",
                "effect_overlaps_combatant_body": "partial",
                "effect_only_frame": False,
            }
        )

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
    assert annotation.explanation_value == "high"
    assert annotation.combat_encounter_kind == "uncertain"
    assert annotation.combat_encounter_basis == "ambiguous"
    assert annotation.selection_coverage_facet is None
    assert diagnostics.attempt_count == 5


def test_positive_ordinary_combat_is_reclassified_by_dedicated_verification() -> None:
    """主推論の通常戦闘誤判定が専用確認で主要戦闘へ訂正されること。

    Arrange:
        - 主推論が掲載可能な戦闘をordinaryと誤判定する応答が用意される
        - 戦闘種別の専用確認がmajorを返す応答が用意される
        - 戦闘可視性と外周構図が掲載可能と確認される
    Act:
        - Candidate Annotation推論が実行される
    Assert:
        - 主推論のordinaryではなく専用確認のmajorが保存されること
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
            return _response(
                _frame_observation_payload(
                    (("frame-a", "battle", "gameplay_action", "high", "hud"),)
                )
            )
        if len(payloads) == 2:
            return _response(
                _combat_encounter_payload(
                    visible=True,
                    evidence="enemy_status_ui",
                    combat_encounter_kind="major",
                    combat_encounter_basis="major_opponent_presentation",
                )
            )
        if len(payloads) == 5:
            return _response(_combat_visibility_edge_audit_payload())
        return _response(
            _combat_visibility_payload(
                opponent_body_visibility="clear",
                opponent_body_framing="complete",
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
    annotation, diagnostics = runtime.annotate_candidate(
        _annotation_request(),
        _catalog(),
        _resolved_model(ModelRole.CANDIDATE_ANNOTATION),
        num_ctx=32768,
    )

    # Assert
    assert annotation.explanation_value == "high"
    assert annotation.combat_encounter_kind == "major"
    assert annotation.combat_encounter_basis == "major_opponent_presentation"
    assert diagnostics.attempt_count == 5
    verification_prompt = _last_message(payloads[1])["content"]
    assert isinstance(verification_prompt, str)
    assert "combat_encounter_kind" in verification_prompt
    assert "積極的な画像内根拠がある場合だけ" in verification_prompt


def test_unconfirmed_combat_scene_has_no_explanation_value() -> None:
    """戦闘sceneで戦闘を確認できないframeが掲載不可にされること。

    Arrange:
        - 戦闘sceneを非戦闘と誤分類した掲載可能応答が用意される
        - 二回の戦闘有無確認で戦闘対象なしとされる
    Act:
        - Candidate Annotation推論が実行される
    Assert:
        - 戦闘sceneとして説明できないためExplanation Valueがnoneにされること
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
            response = _frame_observation_payload(
                (("frame-a", "battle", "gameplay_action", "high", "hud"),)
            )
            observation = _first_frame_observation(response)
            observation["combat_encounter_kind"] = "not_combat"
            observation["combat_encounter_basis"] = "none"
            observation["opponent_body_visibility"] = "absent"
            return _response(response)
        if len(payloads) <= 3:
            return _response(_combat_encounter_payload(visible=False, evidence="none"))
        return _response(_combat_visibility_payload())

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
    assert annotation.explanation_value == "none"
    assert diagnostics.attempt_count == 3
    assert len(payloads) == 3


def test_missed_combat_is_rejected_by_visibility_cross_check() -> None:
    """二回見落とされた戦闘が敵本体の画面端検出で掲載不可にされること。

    Arrange:
        - boss戦を非戦闘と誤分類したrecurring gameplay応答が用意される
        - 戦闘なしが二回返された後、画面端の敵が独立確認で検出される
    Act:
        - Candidate Annotation推論が実行される
    Assert:
        - 戦闘有無確認の見落としにかかわらずExplanation Valueがnoneにされること
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
            response = _frame_observation_payload(
                (("frame-a", "exploration", "gameplay_action", "high", "hud"),)
            )
            observation = _first_frame_observation(response)
            observation["combat_encounter_kind"] = "not_combat"
            observation["combat_encounter_basis"] = "none"
            observation["opponent_body_visibility"] = "absent"
            return _response(response)
        if len(payloads) <= 3:
            return _response(_combat_encounter_payload(visible=False, evidence="none"))
        if len(payloads) == 4:
            return _response(_combat_visibility_payload())
        return _response(
            _combat_visibility_payload(
                opponent_body_visibility="partial",
                opponent_body_framing="edge_cropped",
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
    annotation, diagnostics = runtime.annotate_candidate(
        _annotation_request(),
        _catalog_with_recurring_exploration(),
        _resolved_model(ModelRole.CANDIDATE_ANNOTATION),
        num_ctx=32768,
    )

    # Assert
    assert annotation.explanation_value == "none"
    assert diagnostics.attempt_count == 5
    assert diagnostics.validation_code is None
    assert len(payloads) == 5


def test_negative_combat_encounter_is_confirmed_before_visibility_routing() -> None:
    """一度だけ非戦闘とされたactionが独立確認で戦闘へ戻されること。

    Arrange:
        - 主推論が非戦闘としたrecurring gameplayのactionが用意される
        - 初回確認は非戦闘、独立確認は敵status UIのある戦闘とする応答が用意される
        - 戦闘可視性確認では敵本体が不在とする応答が用意される
    Act:
        - Candidate Annotation推論が実行される
    Assert:
        - 一回の非戦闘判定では確認が終了せず掲載不可にされること
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
            response = _frame_observation_payload(
                (("frame-a", "battle", "gameplay_action", "high", "hud"),)
            )
            observation = _first_frame_observation(response)
            observation["combat_encounter_kind"] = "not_combat"
            observation["combat_encounter_basis"] = "none"
            observation["opponent_body_visibility"] = "absent"
            return _response(response)
        if len(payloads) == 2:
            return _response(_combat_encounter_payload(visible=False, evidence="none"))
        if len(payloads) == 3:
            return _response(
                _combat_encounter_payload(visible=True, evidence="enemy_status_ui")
            )
        return _response(
            {
                "effect_screen_coverage": "over_half",
                "largest_foreground_element": "visual_effect",
                "player_body_visibility": "partial",
                "opponent_body_visibility": "absent",
                "opponent_body_framing": "absent",
                "opponent_presentation": "absent",
                "combat_interaction_visibility": "none",
                "effect_overlaps_combatant_body": "severe",
                "effect_only_frame": False,
            }
        )

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
    assert annotation.explanation_value == "none"
    assert diagnostics.attempt_count == 4
    encounter_schema = payloads[1]["format"]
    assert isinstance(encounter_schema, Mapping)
    encounter_properties = encounter_schema["properties"]
    assert isinstance(encounter_properties, Mapping)
    assert encounter_properties["combat_encounter_basis"]["enum"] == [
        "none",
        "ordinary_opponent_presentation",
        "ordinary_encounter_presentation",
        "major_opponent_presentation",
        "major_encounter_presentation",
        "ambiguous",
    ]
    encounter_prompt = _last_message(payloads[1])["content"]
    assert isinstance(encounter_prompt, str)
    assert "積極的な画像内根拠がある場合だけordinary" in encounter_prompt
    assert "どちらの積極的根拠もない場合はuncertain" in encounter_prompt
    confirmation_prompt = _last_message(payloads[2])["content"]
    assert isinstance(confirmation_prompt, str)
    assert "先の回答を推測せず" in confirmation_prompt
    visibility_prompt = _last_message(payloads[3])["content"]
    assert isinstance(visibility_prompt, str)
    assert "opponent_body_framing" in visibility_prompt


def test_combat_encounter_schema_failure_is_retried() -> None:
    """戦闘有無確認のschema違反が一回だけ再試行されること。

    Arrange:
        - recurring gameplayの曖昧なaction応答が用意される
        - 初回だけ根拠fieldを欠き、再試行で戦闘と確認される応答が用意される
    Act:
        - Candidate Annotation推論が実行される
    Assert:
        - stable validation code付きで修復され、敵可視性確認へ進むこと
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
            response = _frame_observation_payload(
                (("frame-a", "battle", "gameplay_action", "high", "hud"),)
            )
            observation = _first_frame_observation(response)
            observation["combat_encounter_kind"] = "not_combat"
            observation["combat_encounter_basis"] = "none"
            observation["opponent_body_visibility"] = "absent"
            return _response(response)
        if len(payloads) == 2:
            return _response({"combat_encounter_kind": "ordinary"})
        if len(payloads) == 3:
            return _response(
                _combat_encounter_payload(
                    visible=True,
                    evidence="enemy_status_ui",
                )
            )
        if len(payloads) == 6:
            return _response(_combat_visibility_edge_audit_payload())
        return _response(
            {
                "effect_screen_coverage": "under_quarter",
                "largest_foreground_element": "opponent_body",
                "player_body_visibility": "clear",
                "opponent_body_visibility": "clear",
                "opponent_body_framing": "complete",
                "opponent_presentation": "prominent",
                "combat_interaction_visibility": "direct",
                "effect_overlaps_combatant_body": "partial",
                "effect_only_frame": False,
            }
        )

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
    assert annotation.explanation_value == "high"
    assert diagnostics.attempt_count == 6
    assert diagnostics.validation_code == (
        "combat_encounter_verification_schema_invalid"
    )
    third_prompt = _last_message(payloads[2])["content"]
    assert isinstance(third_prompt, str)
    assert "combat_encounter_verification_schema_invalid" in third_prompt
    assert len(payloads) == 6
    edge_audit_prompt = _last_message(payloads[5])["content"]
    assert isinstance(edge_audit_prompt, str)
    assert "外周30%を切り出した診断画像" in edge_audit_prompt
    confirmation_prompt = _last_message(payloads[4])["content"]
    assert isinstance(confirmation_prompt, str)
    assert "掲載可否を確定する独立した再確認" in confirmation_prompt


@pytest.mark.parametrize(
    "invalid_response_kind",
    ("missing_required_field", "contradictory_opponent_signals"),
)
def test_combat_visibility_schema_failure_is_retried(
    invalid_response_kind: str,
) -> None:
    """戦闘可視性専用確認のschema違反が一回だけ再試行されること。

    Arrange:
        - 掲載可能な戦闘応答と、field欠落または相関矛盾を持つ専用応答が用意される
        - 再試行では敵本体が明瞭な有効応答が用意される
    Act:
        - Candidate Annotation推論が実行される
    Assert:
        - stable validation code付きの再試行結果が採用されること
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
            return _response(
                _frame_observation_payload(
                    (("frame-a", "battle", "gameplay_action", "high", "hud"),)
                )
            )
        if len(payloads) == 2:
            return _response(_combat_encounter_payload(visible=True, evidence="both"))
        verification = {
            "effect_screen_coverage": "under_quarter",
            "largest_foreground_element": "opponent_body",
            "player_body_visibility": "clear",
            "opponent_body_visibility": "clear",
            "opponent_body_framing": "complete",
            "opponent_presentation": "prominent",
            "combat_interaction_visibility": "direct",
            "effect_overlaps_combatant_body": "partial",
            "effect_only_frame": False,
        }
        if len(payloads) == 3:
            if invalid_response_kind == "missing_required_field":
                del verification["effect_only_frame"]
            else:
                verification["opponent_body_visibility"] = "absent"
                verification["opponent_body_framing"] = "absent"
        if len(payloads) == 6:
            return _response(_combat_visibility_edge_audit_payload())
        return _response(verification)

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
    assert annotation.explanation_value == "high"
    assert diagnostics.attempt_count == 6
    assert (
        diagnostics.validation_code == "combat_visibility_verification_schema_invalid"
    )
    third_prompt = _last_message(payloads[3])["content"]
    assert isinstance(third_prompt, str)
    assert "combat_visibility_verification_schema_invalid" in third_prompt
    confirmation_prompt = _last_message(payloads[4])["content"]
    assert isinstance(confirmation_prompt, str)
    assert "掲載可否を確定する独立した再確認" in confirmation_prompt


def test_dialogue_and_combat_visibility_are_rechecked_separately() -> None:
    """台詞の修復後に戦闘可視性が専用推論で再確認されること。

    Arrange:
        - 初回に文脈付き映画演出の台詞と掲載可能な戦闘を同時に返す応答が用意される
        - 台詞修復後に敵本体が見えない専用応答が用意される
    Act:
        - Candidate Annotation推論が実行される
    Assert:
        - 台詞と戦闘が別々に再確認され、掲載不可にされること
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
        if len(payloads) == 3:
            return _response(_combat_encounter_payload(visible=True, evidence="both"))
        if len(payloads) == 4:
            return _response(
                {
                    "effect_screen_coverage": "under_quarter",
                    "largest_foreground_element": "player_body",
                    "player_body_visibility": "clear",
                    "opponent_body_visibility": "absent",
                    "opponent_body_framing": "absent",
                    "opponent_presentation": "absent",
                    "combat_interaction_visibility": "none",
                    "effect_overlaps_combatant_body": "partial",
                    "effect_only_frame": False,
                }
            )
        response = _frame_observation_payload(
            (("frame-a", "battle", "event_dialogue", "high", "dialogue"),)
        )
        observation = _first_frame_observation(response)
        observation["cinematic_event_presentation"] = True
        observation["visible_action"] = True
        observation["combat_encounter_kind"] = "ordinary"
        observation["combat_encounter_basis"] = "ordinary_opponent_presentation"
        observation["opponent_body_visibility"] = "clear"
        if len(payloads) == 2:
            observation["on_screen_dialogue_text_visible"] = False
            observation["dialogue_text_presentation"] = "none"
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
        _annotation_request(),
        _catalog(),
        _resolved_model(ModelRole.CANDIDATE_ANNOTATION),
        num_ctx=32768,
    )

    # Assert
    assert annotation.explanation_value == "none"
    assert diagnostics.attempt_count == 4
    assert diagnostics.validation_code == (
        "candidate_annotation_dialogue_visibility_unverified"
    )
    second_prompt = _last_message(payloads[1])["content"]
    assert isinstance(second_prompt, str)
    assert "画面内台詞文字を画像だけに対して再確認" in second_prompt
    third_prompt = _last_message(payloads[2])["content"]
    assert isinstance(third_prompt, str)
    assert "combat_encounter_kind" in third_prompt
    fourth_prompt = _last_message(payloads[3])["content"]
    assert isinstance(fourth_prompt, str)
    assert "この画像1枚に実際に見える画素だけ" in fourth_prompt


def test_relationship_repair_is_followed_by_combat_visibility_check() -> None:
    """関係違反の修復後に戦闘可視性が専用推論で再確認されること。

    Arrange:
        - 初回はSpoiler関係が不正で、敵本体が明瞭とされた戦闘応答が用意される
        - 2回目はSpoiler関係が修正され、専用応答では敵本体なしとされる
    Act:
        - Candidate Annotation推論が実行される
    Assert:
        - 関係修正後の専用推論で掲載不可にされること
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
        if len(payloads) == 2:
            return _response(
                {
                    "frame_observations": [
                        {
                            "frame_id": "frame-a",
                            "spoiler_evidence": "画面内に軽微な進行情報が見える",
                        }
                    ]
                }
            )
        if len(payloads) == 3:
            return _response(_combat_encounter_payload(visible=True, evidence="both"))
        if len(payloads) == 4:
            return _response(
                {
                    "effect_screen_coverage": "over_half",
                    "largest_foreground_element": "visual_effect",
                    "player_body_visibility": "partial",
                    "opponent_body_visibility": "absent",
                    "opponent_body_framing": "absent",
                    "opponent_presentation": "absent",
                    "combat_interaction_visibility": "none",
                    "effect_overlaps_combatant_body": "severe",
                    "effect_only_frame": True,
                }
            )
        response = _frame_observation_payload(
            (("frame-a", "battle", "gameplay_action", "high", "hud"),)
        )
        observation = _first_frame_observation(response)
        observation["spoiler_risk"] = "low"
        observation["spoiler_evidence"] = ""
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
        _annotation_request(),
        _catalog(),
        _resolved_model(ModelRole.CANDIDATE_ANNOTATION),
        num_ctx=32768,
    )

    # Assert
    assert annotation.explanation_value == "none"
    assert diagnostics.attempt_count == 4
    assert diagnostics.validation_code == "candidate_annotation_relationship_invalid"
    second_prompt = _last_message(payloads[1])["content"]
    assert isinstance(second_prompt, str)
    assert "spoiler_riskを変更しません" in second_prompt
    third_prompt = _last_message(payloads[2])["content"]
    assert isinstance(third_prompt, str)
    assert "combat_encounter_kind" in third_prompt
    fourth_prompt = _last_message(payloads[3])["content"]
    assert isinstance(fourth_prompt, str)
    assert "この画像1枚に実際に見える画素だけ" in fourth_prompt
    assert len(payloads) == 4


def test_relationship_repair_does_not_replace_dialogue_visibility_recheck() -> None:
    """関係修復が画面内台詞の独立再確認として数えられないこと。

    Arrange:
        - 文脈付き映画演出の台詞と、修復可能なSpoiler関係違反が返される
        - 関係修復後の独立応答では画面内台詞がないと返される
    Act:
        - Candidate Annotation推論が実行される
    Assert:
        - 関係修復とは別の3回目で台詞が再確認され掲載不可になること
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
        if len(payloads) == 2:
            return _response(
                {
                    "frame_observations": [
                        {
                            "frame_id": "frame-a",
                            "spoiler_evidence": "画面内に軽微な進行情報が見える",
                        }
                    ]
                }
            )
        response = _frame_observation_payload(
            (("frame-a", "exploration", "event_dialogue", "high", "dialogue"),)
        )
        observation = _first_frame_observation(response)
        observation["cinematic_event_presentation"] = True
        observation["spoiler_risk"] = "low"
        observation["spoiler_evidence"] = (
            "" if len(payloads) == 1 else "画面内に軽微な進行情報が見える"
        )
        if len(payloads) == 3:
            observation["on_screen_dialogue_text_visible"] = False
            observation["dialogue_text_presentation"] = "none"
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
        _annotation_request(),
        _catalog(),
        _resolved_model(ModelRole.CANDIDATE_ANNOTATION),
        num_ctx=32768,
    )

    # Assert
    assert annotation.explanation_value == "none"
    assert diagnostics.attempt_count == 3
    assert diagnostics.validation_code == (
        "candidate_annotation_dialogue_visibility_unverified"
    )
    second_prompt = _last_message(payloads[1])["content"]
    assert isinstance(second_prompt, str)
    assert "spoiler_riskを変更しません" in second_prompt
    third_prompt = _last_message(payloads[2])["content"]
    assert isinstance(third_prompt, str)
    assert "画面内台詞文字を画像だけに対して再確認" in third_prompt
    assert len(payloads) == 3


@pytest.mark.parametrize(
    ("transient_transition_effect", "expected_value"),
    ((False, "high"), (True, "none")),
)
def test_map_transition_is_visually_rechecked(
    transient_transition_effect: bool,
    expected_value: str,
) -> None:
    """地図を隠す一時的な移動エフェクトが画像だけで再確認されること。

    Arrange:
        - 掲載価値ありとされた地図応答が用意される
        - 安定した地図または白いwipeを持つ専用応答が用意される
    Act:
        - Candidate Annotation推論が実行される
    Assert:
        - 一時的な遷移がある地図だけ掲載不可にされること
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
            return _response(
                _frame_observation_payload(
                    (("frame-a", "exploration", "map", "high", "menu"),)
                )
            )
        return _response(
            _publication_boundary_payload(
                transient_transition_effect=transient_transition_effect,
                transition_effect_kind=(
                    "white_wipe" if transient_transition_effect else "none"
                ),
                transition_effect_coverage=(
                    "over_half" if transient_transition_effect else "none"
                ),
                primary_content_readability=(
                    "obscured" if transient_transition_effect else "clear"
                ),
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
    annotation, diagnostics = runtime.annotate_candidate(
        _annotation_request(),
        _catalog(),
        _resolved_model(ModelRole.CANDIDATE_ANNOTATION),
        num_ctx=32768,
    )

    # Assert
    assert annotation.explanation_value == expected_value
    assert diagnostics.attempt_count == 2
    assert diagnostics.validation_code is None
    verification_schema = payloads[1]["format"]
    assert isinstance(verification_schema, Mapping)
    verification_properties = verification_schema.get("properties")
    assert isinstance(verification_properties, Mapping)
    assert set(verification_properties) == {
        "transient_transition_effect",
        "transition_effect_kind",
        "transition_effect_coverage",
        "cinematic_letterbox",
        "event_staging",
        "on_screen_dialogue_text_visible",
        "visible_character_action",
        "primary_content_readability",
    }
    second_prompt = _last_message(payloads[1])["content"]
    assert isinstance(second_prompt, str)
    assert "この画像1枚に実際に見える画素だけ" in second_prompt
    assert "白いwipe、太い光帯" in second_prompt
    assert "地図の雲、通常のcursor" in second_prompt
    assert "音声、前後場面、説明文は使いません" in second_prompt


def test_combat_misclassification_does_not_skip_map_transition_verification() -> None:
    """戦闘から非戦闘へ訂正された地図でも掲載境界が確認されること。

    Arrange:
        - 遷移中の地図が掲載可能なordinary戦闘と誤判定される
        - 戦闘有無と可視性の専用確認では非戦闘と確認される
        - 掲載境界専用確認では白いwipeが確認される
    Act:
        - Candidate Annotation推論が実行される
    Assert:
        - 非戦闘への訂正後に掲載境界が確認され掲載不可にされること
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
            response = _frame_observation_payload(
                (("frame-a", "exploration", "map", "high", "menu"),)
            )
            observation = _first_frame_observation(response)
            observation["visible_character_or_enemy"] = True
            observation["combat_encounter_kind"] = "ordinary"
            observation["combat_encounter_basis"] = "ordinary_opponent_presentation"
            observation["player_body_visibility"] = "clear"
            observation["opponent_body_visibility"] = "clear"
            return _response(response)
        if len(payloads) in {2, 3}:
            return _response(_combat_encounter_payload(visible=False, evidence="none"))
        if len(payloads) in {4, 5}:
            return _response(_combat_visibility_payload())
        return _response(
            _publication_boundary_payload(
                transient_transition_effect=True,
                transition_effect_kind="white_wipe",
                transition_effect_coverage="over_half",
                primary_content_readability="obscured",
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
    annotation, diagnostics = runtime.annotate_candidate(
        _annotation_request(),
        _catalog(),
        _resolved_model(ModelRole.CANDIDATE_ANNOTATION),
        num_ctx=32768,
    )

    # Assert
    assert annotation.explanation_value == "none"
    assert annotation.combat_encounter_kind == "not_combat"
    assert annotation.combat_encounter_basis == "none"
    assert diagnostics.attempt_count == 6
    assert len(payloads) == 6
    publication_prompt = _last_message(payloads[5])["content"]
    assert isinstance(publication_prompt, str)
    assert "transient_transition_effect" in publication_prompt


@pytest.mark.parametrize(
    ("content_kind", "dialogue_visible", "expected_value"),
    (("gameplay_idle", False, "none"), ("event_dialogue", True, "high")),
)
def test_static_cinematic_setup_is_visually_rechecked(
    content_kind: str,
    dialogue_visible: bool,
    expected_value: str,
) -> None:
    """台詞も動作もない映画的な人物配置だけが掲載不可にされること。

    Arrange:
        - cinematic sceneの静止場面または台詞場面が用意される
        - 黒帯とevent人物配置を持つ画像だけの専用応答が用意される
    Act:
        - Candidate Annotation推論が実行される
    Assert:
        - 画面内台詞のない静止eventだけ掲載不可にされること
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
            return _response(
                _frame_observation_payload(
                    (
                        (
                            "frame-a",
                            "town",
                            content_kind,
                            "high",
                            "dialogue" if dialogue_visible else "none",
                        ),
                    )
                )
            )
        return _response(
            _publication_boundary_payload(
                cinematic_letterbox=True,
                event_staging=True,
                dialogue_visible=dialogue_visible,
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
    annotation, diagnostics = runtime.annotate_candidate(
        _annotation_request(),
        _catalog_with_cinematic_town(),
        _resolved_model(ModelRole.CANDIDATE_ANNOTATION),
        num_ctx=32768,
    )

    # Assert
    assert annotation.explanation_value == expected_value
    assert diagnostics.attempt_count == 2
    assert diagnostics.validation_code is None


def test_letterboxed_ordinary_scene_is_visually_rechecked() -> None:
    """Otherへ誤分類された上下黒帯の静止eventも専用確認されること。

    Arrange:
        - 上下黒帯の画像が通常playの待機場面と誤分類される応答が用意される
        - 複数人物の固定配置だが黒帯を見落とした専用応答が用意される
    Act:
        - Candidate Annotation推論が実行される
    Assert:
        - 画素から黒帯が補完され、台詞のない静止eventが掲載不可にされること
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
            return _response(
                _frame_observation_payload(
                    (("frame-a", "other", "gameplay_idle", "medium", "none"),)
                )
            )
        return _response(
            _publication_boundary_payload(
                cinematic_letterbox=False,
                event_staging=True,
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
    annotation, diagnostics = runtime.annotate_candidate(
        _annotation_request(image_bytes=_letterboxed_image_bytes()),
        _catalog(),
        _resolved_model(ModelRole.CANDIDATE_ANNOTATION),
        num_ctx=32768,
    )

    # Assert
    assert annotation.explanation_value == "none"
    assert diagnostics.attempt_count == 2
    assert len(payloads) == 2


def test_letterboxed_combat_runs_independent_publication_boundary_check() -> None:
    """戦闘可視性を通過した黒帯画像も掲載境界で独立確認されること。

    Arrange:
        - 敵が明瞭な戦闘として一貫して確認される上下黒帯画像が用意される
        - 掲載境界確認では一時的な遷移effectが観測される
    Act:
        - Candidate Annotation推論が実行される
    Assert:
        - 戦闘確認後にも掲載境界確認が実行され掲載不可になること
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
            return _response(
                _frame_observation_payload(
                    (("frame-a", "battle", "gameplay_action", "high", "hud"),)
                )
            )
        if len(payloads) == 2:
            return _response(_combat_encounter_payload(visible=True, evidence="both"))
        if len(payloads) in {3, 4}:
            return _response(
                _combat_visibility_payload(
                    opponent_body_visibility="clear",
                    opponent_body_framing="complete",
                )
            )
        if len(payloads) == 5:
            return _response(_combat_visibility_edge_audit_payload())
        return _response(
            _publication_boundary_payload(
                transient_transition_effect=True,
                transition_effect_kind="white_wipe",
                transition_effect_coverage="over_half",
                primary_content_readability="obscured",
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
    annotation, diagnostics = runtime.annotate_candidate(
        _annotation_request(image_bytes=_letterboxed_image_bytes()),
        _catalog(),
        _resolved_model(ModelRole.CANDIDATE_ANNOTATION),
        num_ctx=32768,
    )

    # Assert
    assert annotation.explanation_value == "none"
    assert diagnostics.attempt_count == 6
    assert len(payloads) == 6
    assert "transient_transition_effect" in str(_last_message(payloads[5])["content"])


def test_publication_boundary_schema_failure_is_retried() -> None:
    """掲載境界専用確認のschema違反が一回だけ再試行されること。

    Arrange:
        - 掲載価値ありの地図と、必須fieldを欠く専用応答が用意される
        - 再試行では安定した地図を示す有効応答が用意される
    Act:
        - Candidate Annotation推論が実行される
    Assert:
        - stable validation code付きの再試行結果が採用されること
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
            return _response(
                _frame_observation_payload(
                    (("frame-a", "exploration", "map", "high", "menu"),)
                )
            )
        verification = _publication_boundary_payload()
        if len(payloads) == 2:
            del verification["primary_content_readability"]
        return _response(verification)

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
    assert annotation.explanation_value == "high"
    assert diagnostics.attempt_count == 3
    assert diagnostics.validation_code == (
        "publication_boundary_verification_schema_invalid"
    )
    third_prompt = _last_message(payloads[2])["content"]
    assert isinstance(third_prompt, str)
    assert "publication_boundary_verification_schema_invalid" in third_prompt


def test_candidate_relationship_failure_is_repaired_with_explicit_contract() -> None:
    """Spoiler関係違反が分類を凍結した小さいrepair契約で修復されること。

    Arrange:
        - Cueあり入力と、none relevance・low risk・空evidenceの応答が用意される
        - 2回目は不足したSpoiler Evidenceだけを返す応答が用意される
    Act:
        - Candidate Annotation推論が実行される
    Assert:
        - Cue本文を再送せず、riskを変えない部分repairが統合されること
    """
    # Arrange
    request = _annotation_request()
    payloads: list[Mapping[str, object]] = []

    def requester(
        _method: str,
        _url: str,
        payload: Mapping[str, object] | None,
        _timeout: float,
    ) -> object:
        assert payload is not None
        payloads.append(payload)
        if len(payloads) == 2:
            return _response(
                {
                    "frame_observations": [
                        {
                            "frame_id": "frame-a",
                            "spoiler_evidence": "画面内に軽微な進行情報が見える",
                        }
                    ]
                }
            )
        response = _annotation_payload()
        response["context_relevance"] = "none"
        response["supporting_context_cue_ids"] = []
        observation = _first_frame_observation(response)
        observation["spoiler_risk"] = "low"
        observation["spoiler_evidence"] = ""
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
    assert annotation.context_relevance == "none"
    assert annotation.spoiler_risk == "low"
    assert annotation.spoiler_evidence == "画面内に軽微な進行情報が見える"
    assert diagnostics.attempt_count == 2
    assert diagnostics.validation_code == "candidate_annotation_relationship_invalid"
    first_prompt = _last_message(payloads[0])["content"]
    second_prompt = _last_message(payloads[1])["content"]
    assert isinstance(first_prompt, str)
    assert isinstance(second_prompt, str)
    assert "正体を明かす台詞" in first_prompt
    assert "正体を明かす台詞" not in second_prompt
    assert "spoiler_riskがnoneならspoiler_evidenceは空文字列" in first_prompt
    assert "candidate_annotation_relationship_invalid" in second_prompt
    assert "spoiler_riskを変更しません" in second_prompt
    assert '"scene_slug"' not in second_prompt
    second_options = payloads[1]["options"]
    assert isinstance(second_options, dict)
    assert second_options["num_predict"] == 1024
    second_schema = payloads[1]["format"]
    assert isinstance(second_schema, dict)
    assert second_schema["required"] == ["frame_observations"]
    schema_text = json.dumps(second_schema, ensure_ascii=False)
    assert "spoiler_risk" not in schema_text
    evidence_schema = second_schema["properties"]["frame_observations"]["items"][
        "properties"
    ]["spoiler_evidence"]
    assert evidence_schema == {
        "type": "string",
        "minLength": 1,
        "maxLength": 160,
    }


def test_candidate_context_relationship_repairs_only_supporting_cue_ids() -> None:
    """Context Cue関係違反で凍結済みrelevanceの従属IDだけが修復されること。

    Arrange:
        - strong relevanceに空のsupporting Cue IDを返す初回応答が用意される
        - 2回目は入力Cue IDだけを返す部分repair応答が用意される
    Act:
        - Candidate Annotation推論が実行される
    Assert:
        - relevanceとSpoiler分類を変えず、Cue IDだけが統合されること
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
        if len(payloads) == 2:
            return _response({"supporting_context_cue_ids": ["cue-a"]})
        response = _annotation_payload()
        response["context_relevance"] = "strong"
        response["supporting_context_cue_ids"] = []
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
        _annotation_request(),
        _catalog(),
        _resolved_model(ModelRole.CANDIDATE_ANNOTATION),
        num_ctx=32768,
    )

    # Assert
    assert annotation.context_relevance == "strong"
    assert annotation.supporting_context_cue_ids == ("cue-a",)
    assert annotation.spoiler_risk == "high"
    assert annotation.spoiler_evidence == "最終ボスの正体が画面で明示される"
    assert diagnostics.attempt_count == 2
    assert diagnostics.validation_code == "candidate_annotation_relationship_invalid"
    second_schema = payloads[1]["format"]
    assert isinstance(second_schema, dict)
    assert second_schema["required"] == ["supporting_context_cue_ids"]
    schema_text = json.dumps(second_schema, ensure_ascii=False)
    assert "context_relevance" not in schema_text
    assert "spoiler_risk" not in schema_text
    cue_schema = second_schema["properties"]["supporting_context_cue_ids"]
    assert cue_schema["minItems"] == 1
    assert cue_schema["items"]["enum"] == ["cue-a"]
    second_prompt = _last_message(payloads[1])["content"]
    assert isinstance(second_prompt, str)
    assert "context_relevanceを変更しません" in second_prompt
    assert "正体を明かす台詞" in second_prompt


def test_candidate_multiple_relationships_share_one_repair_attempt() -> None:
    """Cue IDとSpoiler Evidenceの違反が同じ1回のrepairで修復されること。

    Arrange:
        - strong relevanceのCue IDとlow riskのevidenceが空の応答が用意される
        - 両方の従属fieldだけを返すrepair応答が用意される
    Act:
        - Candidate Annotation推論が実行される
    Assert:
        - 分類を凍結したまま両方が統合され、追加retryされないこと
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
        if len(payloads) == 2:
            return _response(
                {
                    "supporting_context_cue_ids": ["cue-a"],
                    "frame_observations": [
                        {
                            "frame_id": "frame-a",
                            "spoiler_evidence": "画面内に軽微な進行情報が見える",
                        }
                    ],
                }
            )
        response = _annotation_payload()
        response["context_relevance"] = "strong"
        response["supporting_context_cue_ids"] = []
        observation = _first_frame_observation(response)
        observation["spoiler_risk"] = "low"
        observation["spoiler_evidence"] = ""
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
        _annotation_request(),
        _catalog(),
        _resolved_model(ModelRole.CANDIDATE_ANNOTATION),
        num_ctx=32768,
    )

    # Assert
    assert annotation.context_relevance == "strong"
    assert annotation.supporting_context_cue_ids == ("cue-a",)
    assert annotation.spoiler_risk == "low"
    assert annotation.spoiler_evidence == "画面内に軽微な進行情報が見える"
    assert diagnostics.attempt_count == 2
    assert len(payloads) == 2
    second_schema = payloads[1]["format"]
    assert isinstance(second_schema, dict)
    assert second_schema["required"] == [
        "supporting_context_cue_ids",
        "frame_observations",
    ]


def test_candidate_none_spoiler_risk_repairs_evidence_to_empty() -> None:
    """none riskに付いたevidenceがriskを変えず空文字列へ修復されること。

    Arrange:
        - none riskに非空evidenceを返す初回応答が用意される
        - evidenceだけを空文字列で返すrepair応答が用意される
    Act:
        - Candidate Annotation推論が実行される
    Assert:
        - none分類を維持し、従属evidenceだけが空へ統合されること
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
        if len(payloads) == 2:
            return _response(
                {
                    "frame_observations": [
                        {"frame_id": "frame-a", "spoiler_evidence": ""}
                    ]
                }
            )
        response = _annotation_payload()
        observation = _first_frame_observation(response)
        observation["spoiler_risk"] = "none"
        observation["spoiler_evidence"] = "分類と矛盾する根拠"
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
        _annotation_request(),
        _catalog(),
        _resolved_model(ModelRole.CANDIDATE_ANNOTATION),
        num_ctx=32768,
    )

    # Assert
    assert annotation.spoiler_risk == "none"
    assert annotation.spoiler_evidence == ""
    assert diagnostics.attempt_count == 2
    second_schema = payloads[1]["format"]
    assert isinstance(second_schema, dict)
    evidence_schema = second_schema["properties"]["frame_observations"]["items"][
        "properties"
    ]["spoiler_evidence"]
    assert evidence_schema["maxLength"] == 0


@pytest.mark.parametrize(
    ("failure_kind", "expected_reason", "expected_code"),
    (
        (
            "truncated",
            VisionRuntimeFailureReason.RESPONSE_INVALID,
            "candidate_annotation_relationship_repair_response_truncated",
        ),
        (
            "schema",
            VisionRuntimeFailureReason.SCHEMA_INVALID,
            "candidate_annotation_relationship_repair_schema_invalid",
        ),
        (
            "empty",
            VisionRuntimeFailureReason.SCHEMA_INVALID,
            "candidate_annotation_relationship_repair_schema_invalid",
        ),
        (
            "domain",
            VisionRuntimeFailureReason.DOMAIN_INVALID,
            "candidate_annotation_relationship_repair_domain_invalid",
        ),
    ),
)
def test_candidate_relationship_repair_failure_is_fatal(
    failure_kind: str,
    expected_reason: VisionRuntimeFailureReason,
    expected_code: str,
) -> None:
    """部分repairの打ち切り・空・Schema・domain違反がfatalになること。

    Arrange:
        - low riskに空evidenceを持つ初回応答が用意される
        - 指定された種類で失敗する部分repair応答が用意される
    Act:
        - Candidate Annotation推論が実行される
    Assert:
        - 専用codeとattempt 2でfatalになり、3回目が実行されないこと
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
        if len(payloads) == 2:
            if failure_kind == "truncated":
                response = _response({"frame_observations": []})
                response["done_reason"] = "length"
                return response
            if failure_kind == "schema":
                return _response({"unexpected": []})
            if failure_kind == "empty":
                response = _response({})
                message = response["message"]
                assert isinstance(message, dict)
                message["content"] = ""
                return response
            return _response(
                {
                    "frame_observations": [
                        {"frame_id": "frame-a", "spoiler_evidence": ""}
                    ]
                }
            )
        response = _annotation_payload()
        observation = _first_frame_observation(response)
        observation["spoiler_risk"] = "low"
        observation["spoiler_evidence"] = ""
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
    assert captured.value.reason is expected_reason
    assert captured.value.validation_code == expected_code
    assert captured.value.attempt_count == 2
    assert len(payloads) == 2


@pytest.mark.parametrize(
    "repair_observations",
    (
        ({"frame_id": "frame-a", "spoiler_evidence": "軽微な進行情報"},),
        (
            {"frame_id": "frame-a", "spoiler_evidence": "軽微な進行情報"},
            {"frame_id": "frame-a", "spoiler_evidence": "別の進行情報"},
        ),
        (
            {"frame_id": "frame-a", "spoiler_evidence": "軽微な進行情報"},
            {"frame_id": "frame-x", "spoiler_evidence": "別の進行情報"},
        ),
        (
            {"frame_id": "frame-b", "spoiler_evidence": "別の進行情報"},
            {"frame_id": "frame-a", "spoiler_evidence": "軽微な進行情報"},
        ),
        (
            {"frame_id": "frame-a", "spoiler_evidence": "あ" * 161},
            {"frame_id": "frame-b", "spoiler_evidence": "別の進行情報"},
        ),
    ),
    ids=("missing", "duplicate", "unknown", "reordered", "too-long"),
)
def test_candidate_relationship_repair_rejects_invalid_frame_projection(
    repair_observations: tuple[dict[str, str], ...],
) -> None:
    """欠落・重複・unknown・順序変更・上限超過のrepairが拒否されること。

    Arrange:
        - 2 frameのSpoiler Evidenceがどちらも空の初回応答が用意される
        - 不正なframe projectionを返すrepair応答が用意される
    Act:
        - Candidate Annotation推論が実行される
    Assert:
        - 専用domain codeとattempt 2でfatalになること
    """
    # Arrange
    request = _annotation_request_with_frame_ids(
        ("frame-a", "frame-b"),
        include_context=False,
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
        if len(payloads) == 2:
            return _response(
                {"frame_observations": [dict(item) for item in repair_observations]}
            )
        response = _frame_observation_payload(
            (
                ("frame-a", "exploration", "gameplay_idle", "high", "hud"),
                ("frame-b", "exploration", "gameplay_idle", "medium", "hud"),
            ),
            context_relevance="unavailable",
            supporting_context_cue_ids=(),
        )
        raw_observations = response["frame_observations"]
        assert isinstance(raw_observations, list)
        for observation in raw_observations:
            assert isinstance(observation, dict)
            observation["spoiler_risk"] = "low"
            observation["spoiler_evidence"] = ""
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
            request,
            _catalog(),
            _resolved_model(ModelRole.CANDIDATE_ANNOTATION),
            num_ctx=32768,
        )
    assert captured.value.reason is VisionRuntimeFailureReason.DOMAIN_INVALID
    assert (
        captured.value.validation_code
        == "candidate_annotation_relationship_repair_domain_invalid"
    )
    assert captured.value.attempt_count == 2
    assert len(payloads) == 2


def test_candidate_context_relationship_repair_rejects_unknown_cue_id() -> None:
    """Context Cue repairが入力にないCue IDを拒否すること。

    Arrange:
        - strong relevanceに空のCue IDを返す初回応答が用意される
        - 入力にないCue IDを返すrepair応答が用意される
    Act:
        - Candidate Annotation推論が実行される
    Assert:
        - 専用domain codeとattempt 2でfatalになること
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
        if len(payloads) == 2:
            return _response({"supporting_context_cue_ids": ["cue-unknown"]})
        response = _annotation_payload()
        response["context_relevance"] = "strong"
        response["supporting_context_cue_ids"] = []
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
    assert captured.value.reason is VisionRuntimeFailureReason.DOMAIN_INVALID
    assert (
        captured.value.validation_code
        == "candidate_annotation_relationship_repair_domain_invalid"
    )
    assert captured.value.attempt_count == 2
    assert len(payloads) == 2


def test_candidate_relationship_repair_redacts_verbatim_context_evidence() -> None:
    """Cue選択とEvidenceの同時repair後にもCue本文が逐語安全化されること。

    Arrange:
        - strong relevanceのCue IDとlow riskのevidenceが空の応答が用意される
        - Cue IDとCue本文そのものをevidenceに返すrepair応答が用意される
    Act:
        - Candidate Annotation推論が実行される
    Assert:
        - repair後のevidenceが非逐語表現へ置換され、raw Cue本文が残らないこと
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
        if len(payloads) == 2:
            return _response(
                {
                    "supporting_context_cue_ids": ["cue-a"],
                    "frame_observations": [
                        {
                            "frame_id": "frame-a",
                            "spoiler_evidence": "正体を明かす台詞",
                        }
                    ],
                }
            )
        response = _annotation_payload()
        response["context_relevance"] = "strong"
        response["supporting_context_cue_ids"] = []
        observation = _first_frame_observation(response)
        observation["spoiler_risk"] = "low"
        observation["spoiler_evidence"] = ""
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
        _annotation_request(),
        _catalog(),
        _resolved_model(ModelRole.CANDIDATE_ANNOTATION),
        num_ctx=32768,
    )

    # Assert
    assert annotation.context_relevance == "strong"
    assert annotation.supporting_context_cue_ids == ("cue-a",)
    assert annotation.spoiler_risk == "low"
    assert annotation.spoiler_evidence == "low相当の進行情報を映像から判定"
    assert "正体を明かす台詞" not in annotation.spoiler_evidence
    assert (
        diagnostics.validation_code == "candidate_annotation_verbatim_context_redacted"
    )
    assert len(payloads) == 2
    second_prompt = _last_message(payloads[1])["content"]
    assert isinstance(second_prompt, str)
    assert "Context Cue本文をspoiler_evidenceへ引用しません" in second_prompt


def test_candidate_relationship_repair_requires_fully_schema_valid_draft() -> None:
    """後続frameにSchema違反がある応答へ部分repairが適用されないこと。

    Arrange:
        - 先頭frameにSpoiler関係違反、後続frameに未知fieldを持つ応答が用意される
        - 2回目には全fieldがvalidなCandidate Annotation応答が用意される
    Act:
        - Candidate Annotation推論が実行される
    Assert:
        - 初回全体がSchema違反として扱われ、通常の全体再試行が行われること
    """
    # Arrange
    request = _annotation_request_with_frame_ids(
        ("frame-a", "frame-b"),
        include_context=False,
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
        response = _frame_observation_payload(
            (
                ("frame-a", "exploration", "gameplay_idle", "high", "hud"),
                ("frame-b", "exploration", "gameplay_idle", "medium", "hud"),
            ),
            context_relevance="unavailable",
            supporting_context_cue_ids=(),
        )
        if len(payloads) == 1:
            first_observation = _first_frame_observation(response)
            first_observation["spoiler_risk"] = "low"
            first_observation["spoiler_evidence"] = ""
            raw_observations = response["frame_observations"]
            assert isinstance(raw_observations, list)
            second_observation = raw_observations[1]
            assert isinstance(second_observation, dict)
            second_observation["unexpected"] = True
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
    assert annotation.candidate.identifier == "frame-a"
    assert diagnostics.attempt_count == 2
    assert diagnostics.validation_code == "candidate_annotation_schema_invalid"
    second_schema = payloads[1]["format"]
    assert isinstance(second_schema, dict)
    assert "context_relevance" in second_schema["required"]
    second_prompt = _last_message(payloads[1])["content"]
    assert isinstance(second_prompt, str)
    assert "candidate_annotation_schema_invalid" in second_prompt


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
    cue_text = "画面内テキストのあるイベント。画像内容を示す場面"
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
    ("cue_text", "spoiler_evidence"),
    (
        ("はい", "人物がはいと返事する場面"),
        ("OK", "画面にOK表示が示される場面"),
    ),
)
def test_candidate_allows_ambiguous_one_or_two_character_cue_occurrence(
    cue_text: str,
    spoiler_evidence: str,
) -> None:
    """一般的な1〜2文字Cueの出現が逐語引用と判定されないこと。

    Arrange:
        - 一般的な1〜2文字Cueと、その文字列を含む画像由来evidenceが用意される
    Act:
        - Candidate Annotation推論が実行される
    Assert:
        - 独立生成との区別がつかない短い一致が受理されること
    """
    # Arrange
    request = _annotation_request_with_context_text(cue_text)
    response = _annotation_payload()
    _first_frame_observation(response)["spoiler_evidence"] = spoiler_evidence
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
        _catalog(),
        _resolved_model(ModelRole.CANDIDATE_ANNOTATION),
        num_ctx=32768,
    )

    # Assert
    assert annotation.spoiler_evidence == spoiler_evidence


def test_internal_event_summary_keeps_scene_discriminator() -> None:
    """選定前のevent説明にScene間の意味識別子が保持されること。

    Arrange:
        - 異なるSceneを識別する名前と、画面内文字のあるevent観測が用意される
    Act:
        - Candidate Annotation推論が実行される
    Assert:
        - 公開前の内部説明にはScene識別子が保持されること
    """
    # Arrange
    response = _annotation_payload()
    catalog = _catalog_with_battle_display("キャラクターとの会話")
    runtime = OllamaVisionRuntime(
        "http://localhost:11434",
        timeout_seconds=60.0,
        requester=lambda _method, _url, _payload, _timeout: _response(response),
        sleeper=lambda _seconds: None,
        model_state_resolver=_resolved_artifact,
    )

    # Act
    annotation, _ = runtime.annotate_candidate(
        _annotation_request(),
        catalog,
        _resolved_model(ModelRole.CANDIDATE_ANNOTATION),
        num_ctx=32768,
    )

    # Assert
    assert annotation.summary == "キャラクターとの会話の画面内テキストのあるイベント"


def test_idle_gameplay_summary_does_not_claim_the_player_is_waiting() -> None:
    """通常play画面の内部説明へ未観測の待機状態が断定されないこと。

    Arrange:
        - gameplay idleと分類された高評価frameが用意される
    Act:
        - Candidate Annotation推論が実行される
    Assert:
        - Scene識別子を保ちつつ待機中と断定しない説明が返されること
    """
    # Arrange
    response = _frame_observation_payload(
        (("frame-a", "exploration", "gameplay_idle", "high", "hud"),)
    )
    runtime = OllamaVisionRuntime(
        "http://localhost:11434",
        timeout_seconds=60.0,
        requester=lambda _method, _url, _payload, _timeout: _response(response),
        sleeper=lambda _seconds: None,
        model_state_resolver=_resolved_artifact,
    )

    # Act
    annotation, _ = runtime.annotate_candidate(
        _annotation_request(),
        _catalog(),
        _resolved_model(ModelRole.CANDIDATE_ANNOTATION),
        num_ctx=32768,
    )

    # Assert
    assert annotation.summary == "探索の通常プレイ画面"
    assert "待機" not in annotation.summary


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
                "scene_kind": "exploration",
                "selection_role": "ordinary",
            },
            {
                "slug": "battle",
                "display_name": "戦闘",
                "description": "繰り返される通常戦闘",
                "scene_kind": "combat",
                "selection_role": "recurring_gameplay",
            },
            {
                "slug": "other",
                "display_name": "その他",
                "description": "分類不能",
                "scene_kind": "other",
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
                "scene_catalog_match": True,
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
                "cinematic_event_presentation": False,
                "on_screen_dialogue_text_visible": content_kind == "event_dialogue",
                "dialogue_text_presentation": (
                    "dialogue_box" if content_kind == "event_dialogue" else "none"
                ),
                "visible_action": content_kind in {"gameplay_action", "event_action"},
                "visible_character_or_enemy": content_kind
                not in {"map", "save", "tutorial_help", "title"},
                "combat_encounter_kind": (
                    "ordinary"
                    if scene_slug == "battle"
                    and content_kind in {"gameplay_action", "event_action"}
                    else "not_combat"
                ),
                "combat_encounter_basis": (
                    "ordinary_opponent_presentation"
                    if scene_slug == "battle"
                    and content_kind in {"gameplay_action", "event_action"}
                    else "none"
                ),
                "combat_subject_evidence": {
                    "body_plan": "unknown",
                    "scale": "unknown",
                    "surface": "unknown",
                    "colors": [],
                    "traits": [],
                    "distinctiveness": "unclear",
                },
                "player_body_visibility": (
                    "clear"
                    if content_kind not in {"map", "save", "tutorial_help", "title"}
                    else "absent"
                ),
                "opponent_body_visibility": (
                    "clear"
                    if scene_slug == "battle"
                    and content_kind in {"gameplay_action", "event_action"}
                    else "absent"
                ),
                "effect_only_frame": False,
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


def _publication_boundary_payload(
    *,
    transient_transition_effect: bool = False,
    transition_effect_kind: str = "none",
    transition_effect_coverage: str = "none",
    cinematic_letterbox: bool = False,
    event_staging: bool = False,
    dialogue_visible: bool = False,
    visible_character_action: bool = False,
    primary_content_readability: str = "clear",
) -> dict[str, object]:
    return {
        "transient_transition_effect": transient_transition_effect,
        "transition_effect_kind": transition_effect_kind,
        "transition_effect_coverage": transition_effect_coverage,
        "cinematic_letterbox": cinematic_letterbox,
        "event_staging": event_staging,
        "on_screen_dialogue_text_visible": dialogue_visible,
        "visible_character_action": visible_character_action,
        "primary_content_readability": primary_content_readability,
    }


def _combat_encounter_payload(
    *,
    visible: bool,
    evidence: str,
    combat_encounter_kind: str | None = None,
    combat_encounter_basis: str | None = None,
) -> dict[str, object]:
    resolved_kind = (
        combat_encounter_kind
        if combat_encounter_kind is not None
        else ("ordinary" if visible else "not_combat")
    )
    default_basis = {
        "not_combat": "none",
        "ordinary": "ordinary_opponent_presentation",
        "major": "major_opponent_presentation",
        "uncertain": "ambiguous",
    }[resolved_kind]
    return {
        "combat_encounter_kind": resolved_kind,
        "combat_encounter_basis": (
            combat_encounter_basis
            if combat_encounter_basis is not None
            else default_basis
        ),
        "combat_encounter_evidence": evidence,
    }


def _combat_visibility_payload(
    *,
    opponent_body_visibility: str = "absent",
    opponent_body_framing: str = "absent",
    opponent_presentation: str | None = None,
    combat_interaction_visibility: str | None = None,
) -> dict[str, object]:
    resolved_presentation = opponent_presentation or (
        "prominent"
        if opponent_body_visibility == "clear" and opponent_body_framing == "complete"
        else "absent"
        if opponent_body_visibility == "absent"
        else "weak"
    )
    resolved_interaction = combat_interaction_visibility or (
        "direct" if resolved_presentation == "prominent" else "none"
    )
    return {
        "effect_screen_coverage": "none",
        "largest_foreground_element": "environment",
        "player_body_visibility": "clear",
        "opponent_body_visibility": opponent_body_visibility,
        "opponent_body_framing": opponent_body_framing,
        "opponent_presentation": resolved_presentation,
        "combat_interaction_visibility": resolved_interaction,
        "effect_overlaps_combatant_body": "none",
        "effect_only_frame": False,
    }


def _combat_visibility_edge_audit_payload(
    cropped_edges: tuple[str, ...] = (),
) -> dict[str, object]:
    return {
        "edges": [
            {
                "edge": edge,
                "opponent_body_present": edge in cropped_edges,
                "opponent_body_reaches_outer_edge": edge in cropped_edges,
            }
            for edge in ("top", "bottom", "left", "right")
        ]
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
            SceneCatalogEntry(
                "exploration", "探索", "フィールド探索", "exploration", "ordinary"
            ),
            SceneCatalogEntry(
                "battle",
                "戦闘",
                "繰り返される通常戦闘",
                "combat",
                "recurring_gameplay",
            ),
            SceneCatalogEntry("other", "その他", "分類不能", "other", "ordinary"),
        )
    )


def _catalog_with_cinematic_town() -> SceneCatalog:
    return SceneCatalog(
        (
            SceneCatalogEntry(
                "exploration", "探索", "フィールド探索", "exploration", "ordinary"
            ),
            SceneCatalogEntry(
                "town", "街", "街で起こる会話event", "event", "cinematic"
            ),
            SceneCatalogEntry("other", "その他", "分類不能", "other", "ordinary"),
        )
    )


def _catalog_with_recurring_exploration() -> SceneCatalog:
    return SceneCatalog(
        (
            SceneCatalogEntry(
                "exploration",
                "探索",
                "繰り返されるフィールド探索",
                "exploration",
                "recurring_gameplay",
            ),
            SceneCatalogEntry("battle", "戦闘", "通常戦闘", "combat", "ordinary"),
            SceneCatalogEntry("other", "その他", "分類不能", "other", "ordinary"),
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
                scene.scene_kind,
                scene.selection_role,
            )
            for scene in _catalog().scenes
        )
    )


def _annotation_request(
    *,
    image_bytes: bytes | None = None,
) -> CandidateAnnotationRequest:
    frame = FrameCandidate("frame-a", image_bytes or _image_bytes())
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


def _image_bytes() -> bytes:
    """外周cropにも利用できる20x10 JPEGを返す。"""
    output = io.BytesIO()
    Image.new("RGB", (20, 10), color=(32, 64, 96)).save(output, format="JPEG")
    return output.getvalue()


def _letterboxed_image_bytes() -> bytes:
    """上下10%に黒帯を持つJPEGを返す。"""
    image = Image.new("RGB", (320, 180), color=(72, 104, 136))
    pixels = image.load()
    assert pixels is not None
    for y in (*range(18), *range(162, 180)):
        for x in range(image.width):
            pixels[x, y] = (0, 0, 0)
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=95)
    return output.getvalue()


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
