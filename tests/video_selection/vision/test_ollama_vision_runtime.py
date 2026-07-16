import json
from collections.abc import Mapping
from email.message import Message
from fractions import Fraction
from typing import cast
from urllib.error import HTTPError

import pytest

from src.video_selection.models.candidate_annotation_request import (
    CandidateAnnotationRequest,
)
from src.video_selection.models.candidate_moment import CandidateMoment
from src.video_selection.models.context_cue import ContextCue
from src.video_selection.models.frame_candidate import FrameCandidate
from src.video_selection.models.model_role import ModelRole
from src.video_selection.models.model_runtime_identity import ModelRuntimeIdentity
from src.video_selection.models.model_store_kind import ModelStoreKind
from src.video_selection.models.model_update_status import ModelUpdateStatus
from src.video_selection.models.resolved_model import ResolvedModel
from src.video_selection.models.resolved_model_identity import ResolvedModelIdentity
from src.video_selection.models.scene_catalog import SceneCatalog
from src.video_selection.models.scene_catalog_entry import SceneCatalogEntry
from src.video_selection.models.scene_catalog_request import SceneCatalogRequest
from src.video_selection.models.vision_runtime_error import VisionRuntimeError
from src.video_selection.models.vision_runtime_failure_reason import (
    VisionRuntimeFailureReason,
)
from src.video_selection.vision.ollama_vision_runtime import OllamaVisionRuntime


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
        _url: str,
        payload: Mapping[str, object] | None,
        _timeout: float,
    ) -> object:
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
        response["representative_frame_id"] = "foreign-frame"
        return _response(response)

    runtime = OllamaVisionRuntime(
        "http://localhost:11434",
        timeout_seconds=60.0,
        requester=requester,
        sleeper=lambda _seconds: None,
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
    assert captured.value.validation_code == "candidate_annotation_domain_invalid"
    assert "foreign-frame" not in str(captured.value)


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
    response["spoiler_risk"] = "none"
    response["spoiler_evidence"] = ""
    runtime = OllamaVisionRuntime(
        "http://localhost:11434",
        timeout_seconds=60.0,
        requester=lambda _method, _url, _payload, _timeout: _response(response),
        sleeper=lambda _seconds: None,
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
    return {
        "representative_frame_id": "frame-a",
        "scene_slug": "battle",
        "blog_image_type": "event",
        "explanation_value": "high",
        "annotation_summary": "終盤の重要な対決",
        "frame_choice_reason": "対決する人物が明確に写る",
        "screen_text_kind": "dialogue",
        "context_relevance": "strong",
        "supporting_context_cue_ids": ["cue-a"],
        "spoiler_risk": "high",
        "spoiler_evidence": "最終ボスの正体が画面で明示される",
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


def _first_message(payload: Mapping[str, object]) -> Mapping[str, object]:
    messages = payload.get("messages")
    assert isinstance(messages, list)
    assert messages
    message = messages[0]
    assert isinstance(message, dict)
    return cast(dict[str, object], message)
