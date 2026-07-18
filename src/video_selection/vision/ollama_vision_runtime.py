"""Ollama structured outputsを使うVisionRuntime adapter。"""

import base64
import copy
import hashlib
import json
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from fractions import Fraction
from functools import partial
from typing import Literal, TypeVar, cast
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from ..model_runtime.ollama_model_store import OllamaModelStore
from ..models.candidate_annotation import (
    BLOG_IMAGE_TYPES,
    CONTEXT_CUE_RELEVANCES,
    EXPLANATION_VALUES,
    SCREEN_TEXT_KINDS,
    SPOILER_RISKS,
    CandidateAnnotation,
    candidate_annotation_context_is_valid,
    candidate_annotation_relationships_are_valid,
    privacy_safe_candidate_text,
)
from ..models.candidate_annotation_request import CandidateAnnotationRequest
from ..models.model_artifact import ModelArtifact
from ..models.model_artifact_invalid_error import ModelArtifactInvalidError
from ..models.model_role import ModelRole
from ..models.model_store_http_error import ModelStoreHttpError
from ..models.model_store_kind import ModelStoreKind
from ..models.model_store_unavailable_error import ModelStoreUnavailableError
from ..models.resolved_model import ResolvedModel
from ..models.scene_catalog import SceneCatalog
from ..models.scene_catalog_entry import (
    SCENE_SELECTION_ROLES,
    SceneCatalogEntry,
    SceneSelectionRole,
    is_valid_scene_slug,
)
from ..models.scene_catalog_request import SceneCatalogRequest
from ..models.vision_inference_diagnostics import VisionInferenceDiagnostics
from ..models.vision_runtime_error import VisionRuntimeError
from ..models.vision_runtime_failure_reason import VisionRuntimeFailureReason
from ..services.gpu_work_coordinator import GpuWorkCoordinator
from ..utils.http_retry_delay import http_retry_delay
from .vision_contract import (
    CANDIDATE_ANNOTATION_PROMPT_VERSION,
    CANDIDATE_ANNOTATION_SCHEMA,
    CANDIDATE_ANNOTATION_SCHEMA_VERSION,
    CANDIDATE_ANNOTATION_STAGE_CONTRACT_VERSION,
    RETRY_POLICY_VERSION,
    SCENE_CATALOG_PROMPT_VERSION,
    SCENE_CATALOG_SCHEMA,
    SCENE_CATALOG_SCHEMA_VERSION,
    SCENE_CATALOG_STAGE_CONTRACT_VERSION,
)

JsonRequester = Callable[
    [str, str, Mapping[str, object] | None, float],
    object,
]
Sleeper = Callable[[float], None]
ModelStateResolver = Callable[[ResolvedModel], ModelArtifact]
InferenceValue = TypeVar("InferenceValue")
InferenceParser = Callable[[Mapping[str, object]], InferenceValue]
StageKind = Literal["scene_catalog", "candidate_annotation"]

_SCENE_ENTRY_KEYS = {"slug", "display_name", "description", "selection_role"}
_ANNOTATION_KEYS = {
    "representative_frame_id",
    "scene_slug",
    "blog_image_type",
    "explanation_value",
    "annotation_summary",
    "frame_choice_reason",
    "screen_text_kind",
    "context_relevance",
    "supporting_context_cue_ids",
    "spoiler_risk",
    "spoiler_evidence",
}
_RETRYABLE_REASONS = {
    VisionRuntimeFailureReason.TRANSPORT_FAILURE,
    VisionRuntimeFailureReason.RESPONSE_INVALID,
    VisionRuntimeFailureReason.SCHEMA_INVALID,
    VisionRuntimeFailureReason.DOMAIN_INVALID,
}
_PROMPT_REPAIR_REASONS = {
    VisionRuntimeFailureReason.RESPONSE_INVALID,
    VisionRuntimeFailureReason.SCHEMA_INVALID,
    VisionRuntimeFailureReason.DOMAIN_INVALID,
}
_SCENE_CATALOG_SEMANTICS = (
    "selection_roleはordinary=通常の単発scene、cinematic=会話・演出・eventが主体、"
    "recurring_gameplay=戦闘UI・探索・puzzleなど繰り返し現れるplay構造です。"
    "同じ画面構造を一時的な敵やエフェクトだけで別sceneへ分割しません。"
    "sceneはブログで役割が異なる視覚・内容のまとまりとして作ります。\n"
)
_CANDIDATE_ANNOTATION_SEMANTICS = (
    "代表frameは主対象や行動が判別できるframeを優先します。"
    "大きな発光やエフェクトで主対象が隠れるframe、白飛び、移動・画面切替の"
    "途中は避けます。全frameで主対象が判別できない場合は最も明瞭なframeを"
    "選び、explanation_valueをnoneにします。\n"
    "blog_image_typeはnormal_gameplay=探索・戦闘・puzzleなどplayが主体、"
    "event=会話・cutscene・scripted presentationが主体、menu=inventory・装備・"
    "map・設定・shop・save・helpなどinterfaceが主体、title=title・logo・landing、"
    "other=どれにも当てはまらない有効画像です。\n"
    "explanation_valueのnone=主対象や出来事を説明できずブログ掲載価値がない、"
    "low=判別できるが汎用的・重複的、medium=具体的なplay状態や出来事を説明できる、"
    "high=重要な主対象・行動・関係が明瞭で本文を直接補強する、です。\n"
    "context_relevanceのnone=近接していても画像説明と無関係、weak=補足になる、"
    "context_relevanceのstrong=画像の意味を特定するため不可欠、です。"
    "単にContext Cueが存在するだけで"
    "strongにしません。\n"
    "spoiler_riskはnone=汎用的な探索・戦闘、low=軽微な進行情報、medium=固有boss・"
    "終盤固有area・重要quest結果、spoiler_riskのhigh=ending・最終bossの正体や形態・"
    "主要人物の生死・裏切り・犯人や真の正体・中心的な種明かしです。"
    "進行位置だけではriskを上げません。\n"
)


class OllamaVisionRuntime:
    """Scene CatalogとCandidate Annotationの全推論規則を閉じ込める。"""

    def __init__(
        self,
        host: str,
        *,
        timeout_seconds: float,
        requester: JsonRequester | None = None,
        sleeper: Sleeper = time.sleep,
        model_state_resolver: ModelStateResolver | None = None,
        gpu_coordinator: GpuWorkCoordinator | None = None,
    ) -> None:
        if not host.strip() or timeout_seconds <= 0:
            raise ValueError("Ollama VisionRuntimeの接続設定が不正です")
        self._host = host.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._requester = requester or _request_json
        self._sleeper = sleeper
        self._model_store = OllamaModelStore(
            self._host,
            timeout_seconds=self._timeout_seconds,
            requester=self._requester,
        )
        self._model_state_resolver = (
            model_state_resolver or self._resolve_current_model_state
        )
        self._gpu_coordinator = gpu_coordinator

    def create_scene_catalog(
        self,
        request: SceneCatalogRequest,
        model: ResolvedModel,
        *,
        num_ctx: int,
    ) -> tuple[SceneCatalog, VisionInferenceDiagnostics]:
        """共有Scene Catalogをstrict schemaとdomain validationで生成する。"""
        _require_model_role(model, ModelRole.SCENE_CATALOG, num_ctx)
        semantic_input = _scene_catalog_semantic_input(request, model, num_ctx)
        return self._infer(
            stage_kind="scene_catalog",
            request_fingerprint=_fingerprint(semantic_input),
            payload=_scene_catalog_payload(request, model, num_ctx),
            parser=_parse_scene_catalog,
            model=model,
            image_count=len(request.representatives),
            context_cue_count=0,
        )

    def annotate_candidate(
        self,
        request: CandidateAnnotationRequest,
        catalog: SceneCatalog,
        model: ResolvedModel,
        *,
        num_ctx: int,
    ) -> tuple[CandidateAnnotation, VisionInferenceDiagnostics]:
        """一つのCandidate Momentをstrict schemaと所属検証で評価する。"""
        _require_model_role(model, ModelRole.CANDIDATE_ANNOTATION, num_ctx)
        semantic_input = _candidate_semantic_input(request, catalog, model, num_ctx)
        (annotation, free_text_redacted), diagnostics = self._infer(
            stage_kind="candidate_annotation",
            request_fingerprint=_fingerprint(semantic_input),
            payload=_candidate_payload(request, catalog, model, num_ctx),
            parser=lambda value: _parse_candidate_annotation(value, request, catalog),
            model=model,
            image_count=len(request.frame_candidates),
            context_cue_count=len(request.context_cues),
        )
        if free_text_redacted:
            diagnostics = replace(
                diagnostics,
                validation_code="candidate_annotation_verbatim_context_redacted",
            )
        return annotation, diagnostics

    def _infer(
        self,
        *,
        stage_kind: StageKind,
        request_fingerprint: str,
        payload: dict[str, object],
        parser: InferenceParser[InferenceValue],
        model: ResolvedModel,
        image_count: int,
        context_cue_count: int,
    ) -> tuple[InferenceValue, VisionInferenceDiagnostics]:
        """同じsemantic入力を最大2回実行しsafe diagnosticsを返す。"""
        started_at = time.monotonic()
        previous_validation_code: str | None = None
        repair_code: str | None = None
        for attempt in (1, 2):
            attempt_payload = _with_repair_code(payload, repair_code)
            try:
                self._require_frozen_model_state(model)
                response = self._request(attempt_payload)
                self._require_frozen_model_state(model)
                value = parser(_decode_content(response, stage_kind))
            except VisionRuntimeError as error:
                if attempt == 2 or error.reason not in _RETRYABLE_REASONS:
                    raise VisionRuntimeError(
                        error.reason,
                        validation_code=error.validation_code,
                        attempt_count=attempt,
                    ) from None
                previous_validation_code = error.validation_code
                repair_code = _repair_validation_code(error)
                self._sleeper(error.retry_after_seconds)
                continue
            diagnostics = _diagnostics(
                response=response,
                stage_kind=stage_kind,
                request_fingerprint=request_fingerprint,
                model=model,
                attempt_count=attempt,
                validation_code=previous_validation_code,
                image_count=image_count,
                context_cue_count=context_cue_count,
                duration_seconds=time.monotonic() - started_at,
            )
            return value, diagnostics
        raise AssertionError("VisionRuntime retry loop did not terminate")

    def _require_frozen_model_state(self, model: ResolvedModel) -> None:
        """推論前後のmodel artifactがfreeze済みstateと一致することを要求する。"""
        try:
            current = self._model_state_resolver(model)
        except VisionRuntimeError:
            raise
        except Exception:
            raise VisionRuntimeError(
                VisionRuntimeFailureReason.TRANSPORT_FAILURE,
                validation_code="ollama_transport_failure",
            ) from None
        if current.identity != model.execution_identity:
            raise VisionRuntimeError(
                VisionRuntimeFailureReason.MODEL_UNAVAILABLE,
                validation_code="ollama_model_identity_changed",
            )
        if current.runtime_identity != model.runtime_identity:
            raise VisionRuntimeError(
                VisionRuntimeFailureReason.MODEL_UNAVAILABLE,
                validation_code="ollama_runtime_identity_changed",
            )

    def _resolve_current_model_state(self, model: ResolvedModel) -> ModelArtifact:
        """Model Storeのartifact確認portをVision failureへ変換する。"""
        try:
            artifact = self._model_store.resolve_current_artifact(model.configured_name)
        except ModelArtifactInvalidError:
            raise VisionRuntimeError(
                VisionRuntimeFailureReason.RESPONSE_INVALID,
                validation_code="ollama_model_identity_response_invalid",
            ) from None
        except ModelStoreHttpError as error:
            raise _http_failure(
                error.status_code,
                error.retry_after_seconds,
            ) from None
        except ModelStoreUnavailableError:
            raise VisionRuntimeError(
                VisionRuntimeFailureReason.TRANSPORT_FAILURE,
                validation_code="ollama_transport_failure",
            ) from None
        if artifact is None:
            raise VisionRuntimeError(
                VisionRuntimeFailureReason.MODEL_UNAVAILABLE,
                validation_code="ollama_model_identity_unavailable",
            )
        return artifact

    def _request(self, payload: Mapping[str, object]) -> Mapping[str, object]:
        """transport detailをstable failureへ変換する。"""
        try:
            request = partial(
                self._requester,
                "POST",
                f"{self._host}/api/chat",
                payload,
                self._timeout_seconds,
            )
            response = (
                request()
                if self._gpu_coordinator is None
                else self._gpu_coordinator.run("vision_inference", request)
            )
        except HTTPError as error:
            retry_after = (
                error.headers.get("Retry-After") if error.headers is not None else None
            )
            raise _http_failure(
                error.code,
                http_retry_delay(error.code, retry_after),
            ) from None
        except Exception:
            raise VisionRuntimeError(
                VisionRuntimeFailureReason.TRANSPORT_FAILURE,
                validation_code="ollama_transport_failure",
            ) from None
        if not isinstance(response, dict) or not all(
            isinstance(key, str) for key in response
        ):
            raise VisionRuntimeError(
                VisionRuntimeFailureReason.RESPONSE_INVALID,
                validation_code="ollama_response_invalid",
            )
        return cast(dict[str, object], response)


def _request_json(
    method: str,
    url: str,
    payload: Mapping[str, object] | None,
    timeout: float,
) -> object:
    body = None if payload is None else json.dumps(payload).encode()
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _scene_catalog_payload(
    request: SceneCatalogRequest,
    model: ResolvedModel,
    num_ctx: int,
) -> dict[str, object]:
    hint = request.scene_hint or "なし"
    content = (
        "Video Set全体で共有するブログ画像用Scene Catalogを作成してください。"
        "3〜8 sceneにotherを必ず1件含め、otherのselection_roleはordinaryにします。"
        "画像品質、最終score、採否、推論過程は出力しません。\n"
        + _SCENE_CATALOG_SEMANTICS
        + f"Selection Intent: {request.selection_intent}\nScene Hint: {hint}"
    )
    return {
        "model": model.configured_name,
        "stream": False,
        "think": False,
        "format": SCENE_CATALOG_SCHEMA,
        "options": {"temperature": 0, "num_ctx": num_ctx},
        "messages": [
            {
                "role": "user",
                "content": content,
                "images": [
                    base64.b64encode(item.image_bytes).decode()
                    for item in request.representatives
                ],
            }
        ],
    }


def _candidate_payload(
    request: CandidateAnnotationRequest,
    catalog: SceneCatalog,
    model: ResolvedModel,
    num_ctx: int,
) -> dict[str, object]:
    semantic_request = {
        "candidate_moment_id": request.moment.identifier,
        "frame_candidate_ids": [item.identifier for item in request.frame_candidates],
        "scene_catalog": [_scene_value(item) for item in catalog.scenes],
        "context_cues": [
            {
                "id": cue.identifier,
                "start": _fraction_value(cue.start),
                "end": _fraction_value(cue.end),
                "text": cue.text,
            }
            for cue in request.context_cues
        ],
        "video_set_progress": _fraction_value(request.video_set_progress),
        "selection_intent": request.selection_intent,
    }
    content = (
        "入力された1〜3枚からブログ上の意味を最も表すframeを1枚選び、"
        "共有Scene Catalogを使ってCandidate Annotationを返してください。"
        + _CANDIDATE_ANNOTATION_SEMANTICS
        + "画像品質、confidence、final score、eligible、selected、逐語的画面文、"
        "推論過程は出力しません。Context Cue本文をannotation summary、"
        "frame choice reason、spoiler evidenceへ引用しません。正規化後3〜5文字の"
        "Cueは全文、6文字以上のCueは6文字以上の連続部分も自由文へ再出力しません。"
        "representative_frame_idはframe_candidate_idsから、scene_slugは"
        "scene_catalogから選びます。context_cuesが空ならcontext_relevanceは"
        "unavailable、supporting_context_cue_idsは空配列にします。context_cuesが"
        "ある場合はcontext_relevanceをunavailableにせず、weakまたはstrongなら"
        "supporting_context_cue_idsへ入力内のIDを1件以上入れ、noneなら空配列に"
        "します。spoiler_riskがnoneならspoiler_evidenceは空文字列にし、それ以外"
        "なら根拠を空でない自分の言葉で記述します。\n"
        + json.dumps(semantic_request, ensure_ascii=False, sort_keys=True)
    )
    return {
        "model": model.configured_name,
        "stream": False,
        "think": False,
        "format": _candidate_schema(request, catalog),
        "options": {"temperature": 0, "num_ctx": num_ctx},
        "messages": [
            {
                "role": "user",
                "content": content,
                "images": [
                    base64.b64encode(item.image_bytes).decode()
                    for item in request.frame_candidates
                ],
            }
        ],
    }


def _candidate_schema(
    request: CandidateAnnotationRequest,
    catalog: SceneCatalog,
) -> dict[str, object]:
    """requestで選択可能なIDとContext relevanceへschemaを限定する。"""
    schema = copy.deepcopy(CANDIDATE_ANNOTATION_SCHEMA)
    properties = cast(dict[str, dict[str, object]], schema["properties"])
    properties["representative_frame_id"]["enum"] = [
        item.identifier for item in request.frame_candidates
    ]
    properties["scene_slug"]["enum"] = list(catalog.slugs)
    cue_ids = [item.identifier for item in request.context_cues]
    relevance = properties["context_relevance"]
    supporting_cues = properties["supporting_context_cue_ids"]
    if cue_ids:
        relevance["enum"] = ["none", "weak", "strong"]
        items = cast(dict[str, object], supporting_cues["items"])
        items["enum"] = cue_ids
        supporting_cues["maxItems"] = len(cue_ids)
    else:
        relevance["enum"] = ["unavailable"]
        supporting_cues["maxItems"] = 0
    return schema


def _with_repair_code(
    payload: dict[str, object], validation_code: str | None
) -> dict[str, object]:
    if validation_code is None:
        return payload
    copied = cast(dict[str, object], json.loads(json.dumps(payload)))
    messages = cast(list[dict[str, object]], copied["messages"])
    content = cast(str, messages[0]["content"])
    repair = f"前回の出力を修正してください。validation_code={validation_code}"
    if validation_code == "candidate_annotation_relationship_invalid":
        repair += (
            "\n関係を必ず修正します。spoiler_riskがnoneならspoiler_evidenceは"
            "空文字列、low・medium・highならspoiler_evidenceは画面から判断した"
            "根拠を1文以上記述します。context_relevanceがnoneまたはunavailable"
            "ならsupporting_context_cue_idsは空配列、weakまたはstrongなら入力内IDを"
            "1件以上入れます。"
        )
    messages[0]["content"] = f"{content}\n{repair}"
    return copied


def _decode_content(
    response: Mapping[str, object], stage_kind: StageKind
) -> Mapping[str, object]:
    done_reason = response.get("done_reason")
    if response.get("done") is not True or done_reason not in (None, "stop"):
        raise VisionRuntimeError(
            VisionRuntimeFailureReason.RESPONSE_INVALID,
            validation_code=f"{stage_kind}_response_truncated",
        )
    message = response.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise VisionRuntimeError(
            VisionRuntimeFailureReason.RESPONSE_INVALID,
            validation_code=f"{stage_kind}_response_empty",
        )
    try:
        parsed: object = json.loads(content)
    except json.JSONDecodeError:
        raise VisionRuntimeError(
            VisionRuntimeFailureReason.SCHEMA_INVALID,
            validation_code=f"{stage_kind}_schema_invalid",
        ) from None
    if not isinstance(parsed, dict) or not all(isinstance(key, str) for key in parsed):
        raise VisionRuntimeError(
            VisionRuntimeFailureReason.SCHEMA_INVALID,
            validation_code=f"{stage_kind}_schema_invalid",
        )
    return cast(dict[str, object], parsed)


def _parse_scene_catalog(value: Mapping[str, object]) -> SceneCatalog:
    scenes = value.get("scenes")
    if (
        set(value) != {"scenes"}
        or not isinstance(scenes, list)
        or not 3 <= len(scenes) <= 8
    ):
        raise _schema_error("scene_catalog_schema_invalid")
    entries: list[SceneCatalogEntry] = []
    for raw_scene in scenes:
        if not isinstance(raw_scene, dict) or set(raw_scene) != _SCENE_ENTRY_KEYS:
            raise _schema_error("scene_catalog_schema_invalid")
        slug = raw_scene.get("slug")
        display_name = raw_scene.get("display_name")
        description = raw_scene.get("description")
        selection_role = raw_scene.get("selection_role")
        if (
            not isinstance(slug, str)
            or not is_valid_scene_slug(slug)
            or not isinstance(display_name, str)
            or not display_name.strip()
            or not isinstance(description, str)
            or not description.strip()
            or selection_role not in SCENE_SELECTION_ROLES
        ):
            raise _schema_error("scene_catalog_schema_invalid")
        entries.append(
            SceneCatalogEntry(
                slug=slug,
                display_name=display_name,
                description=description,
                selection_role=cast(SceneSelectionRole, selection_role),
            )
        )
    try:
        return SceneCatalog(tuple(entries))
    except ValueError:
        raise _domain_error("scene_catalog_domain_invalid") from None


def _parse_candidate_annotation(
    value: Mapping[str, object],
    request: CandidateAnnotationRequest,
    catalog: SceneCatalog,
) -> tuple[CandidateAnnotation, bool]:
    if set(value) != _ANNOTATION_KEYS:
        raise _schema_error("candidate_annotation_schema_invalid")
    representative_frame_id = value.get("representative_frame_id")
    scene_slug = value.get("scene_slug")
    blog_image_type = value.get("blog_image_type")
    explanation_value = value.get("explanation_value")
    annotation_summary = value.get("annotation_summary")
    frame_choice_reason = value.get("frame_choice_reason")
    screen_text_kind = value.get("screen_text_kind")
    context_relevance = value.get("context_relevance")
    cue_ids = value.get("supporting_context_cue_ids")
    spoiler_risk = value.get("spoiler_risk")
    spoiler_evidence = value.get("spoiler_evidence")
    if (
        not isinstance(representative_frame_id, str)
        or not isinstance(scene_slug, str)
        or blog_image_type not in BLOG_IMAGE_TYPES
        or explanation_value not in EXPLANATION_VALUES
        or not isinstance(annotation_summary, str)
        or not annotation_summary.strip()
        or not isinstance(frame_choice_reason, str)
        or not frame_choice_reason.strip()
        or screen_text_kind not in SCREEN_TEXT_KINDS
        or context_relevance not in CONTEXT_CUE_RELEVANCES
        or not isinstance(cue_ids, list)
        or not all(isinstance(item, str) for item in cue_ids)
        or len(cue_ids) != len(set(cue_ids))
        or spoiler_risk not in SPOILER_RISKS
        or not isinstance(spoiler_evidence, str)
    ):
        raise _schema_error("candidate_annotation_schema_invalid")
    frames = {item.identifier: item for item in request.frame_candidates}
    typed_context_relevance = context_relevance
    typed_cue_ids = tuple(cast(list[str], cue_ids))
    typed_spoiler_risk = spoiler_risk
    available_cue_ids = tuple(item.identifier for item in request.context_cues)
    if representative_frame_id not in frames:
        raise _domain_error("candidate_annotation_representative_frame_unknown")
    if scene_slug not in catalog.slugs:
        raise _domain_error("candidate_annotation_scene_slug_unknown")
    if not candidate_annotation_relationships_are_valid(
        typed_context_relevance,
        typed_cue_ids,
        typed_spoiler_risk,
        spoiler_evidence,
    ):
        raise _domain_error("candidate_annotation_relationship_invalid")
    if not candidate_annotation_context_is_valid(
        typed_context_relevance,
        typed_cue_ids,
        available_cue_ids,
    ):
        raise _domain_error("candidate_annotation_context_invalid")
    annotation_summary, frame_choice_reason, spoiler_evidence, free_text_redacted = (
        _privacy_safe_candidate_texts(
            annotation_summary=annotation_summary,
            frame_choice_reason=frame_choice_reason,
            spoiler_evidence=spoiler_evidence,
            scene_slug=scene_slug,
            blog_image_type=cast(str, blog_image_type),
            spoiler_risk=cast(str, typed_spoiler_risk),
            raw_context_texts=tuple(item.text for item in request.context_cues),
            catalog=catalog,
        )
    )
    try:
        return (
            CandidateAnnotation(
                candidate=frames[representative_frame_id],
                summary=annotation_summary,
                candidate_moment_id=request.moment.identifier,
                scene_slug=scene_slug,
                blog_image_type=blog_image_type,
                explanation_value=explanation_value,
                frame_choice_reason=frame_choice_reason,
                screen_text_kind=screen_text_kind,
                context_relevance=typed_context_relevance,
                supporting_context_cue_ids=typed_cue_ids,
                spoiler_risk=typed_spoiler_risk,
                spoiler_evidence=spoiler_evidence,
            ),
            free_text_redacted,
        )
    except ValueError:
        raise _domain_error("candidate_annotation_domain_invalid") from None


def _privacy_safe_candidate_texts(
    *,
    annotation_summary: str,
    frame_choice_reason: str,
    spoiler_evidence: str,
    scene_slug: str,
    blog_image_type: str,
    spoiler_risk: str,
    raw_context_texts: tuple[str, ...],
    catalog: SceneCatalog,
) -> tuple[str, str, str, bool]:
    """Cue逐語一致fieldだけを視覚・enum由来の安全な説明へ置換する。"""
    scene = next(item for item in catalog.scenes if item.slug == scene_slug)
    summary, summary_redacted = privacy_safe_candidate_text(
        annotation_summary,
        f"{scene.display_name}に分類される{blog_image_type}の場面",
        raw_context_texts,
    )
    reason, reason_redacted = privacy_safe_candidate_text(
        frame_choice_reason,
        f"{scene.description}を視覚的に表すフレーム",
        raw_context_texts,
    )
    evidence, evidence_redacted = privacy_safe_candidate_text(
        spoiler_evidence,
        (
            ""
            if spoiler_risk == "none"
            else f"{spoiler_risk}相当の進行情報を映像から判定"
        ),
        raw_context_texts,
    )
    return (
        summary,
        reason,
        evidence,
        summary_redacted or reason_redacted or evidence_redacted,
    )


def _schema_error(code: str) -> VisionRuntimeError:
    return VisionRuntimeError(
        VisionRuntimeFailureReason.SCHEMA_INVALID,
        validation_code=code,
    )


def _domain_error(code: str) -> VisionRuntimeError:
    return VisionRuntimeError(
        VisionRuntimeFailureReason.DOMAIN_INVALID,
        validation_code=code,
    )


def _scene_catalog_semantic_input(
    request: SceneCatalogRequest,
    model: ResolvedModel,
    num_ctx: int,
) -> dict[str, object]:
    return {
        "representatives": [
            {
                "id": item.identifier,
                "image_sha256": hashlib.sha256(item.image_bytes).hexdigest(),
            }
            for item in request.representatives
        ],
        "selection_intent": request.selection_intent,
        "scene_hint": request.scene_hint,
        "model": {**model.semantic_input(), "num_ctx": num_ctx},
        "generation_options": {"temperature": 0, "stream": False, "think": False},
        "prompt_version": SCENE_CATALOG_PROMPT_VERSION,
        "schema_version": SCENE_CATALOG_SCHEMA_VERSION,
        "stage_contract_version": SCENE_CATALOG_STAGE_CONTRACT_VERSION,
        "retry_policy_version": RETRY_POLICY_VERSION,
    }


def _candidate_semantic_input(
    request: CandidateAnnotationRequest,
    catalog: SceneCatalog,
    model: ResolvedModel,
    num_ctx: int,
) -> dict[str, object]:
    return {
        "candidate_moment_id": request.moment.identifier,
        "frame_candidates": [
            {
                "id": item.identifier,
                "image_sha256": hashlib.sha256(item.image_bytes).hexdigest(),
            }
            for item in request.frame_candidates
        ],
        "context_cues": [
            {
                "id": cue.identifier,
                "text_sha256": hashlib.sha256(cue.text.encode()).hexdigest(),
                "start": _fraction_value(cue.start),
                "end": _fraction_value(cue.end),
            }
            for cue in request.context_cues
        ],
        "cue_selection_policy_version": request.cue_selection_policy_version,
        "scene_catalog": [_scene_value(item) for item in catalog.scenes],
        "video_set_progress": _fraction_value(request.video_set_progress),
        "selection_intent": request.selection_intent,
        "model": {**model.semantic_input(), "num_ctx": num_ctx},
        "generation_options": {"temperature": 0, "stream": False, "think": False},
        "prompt_version": CANDIDATE_ANNOTATION_PROMPT_VERSION,
        "schema_version": CANDIDATE_ANNOTATION_SCHEMA_VERSION,
        "stage_contract_version": CANDIDATE_ANNOTATION_STAGE_CONTRACT_VERSION,
        "retry_policy_version": RETRY_POLICY_VERSION,
    }


def _fingerprint(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _scene_value(scene: SceneCatalogEntry) -> dict[str, str]:
    return {
        "slug": scene.slug,
        "display_name": scene.display_name,
        "description": scene.description,
        "selection_role": scene.selection_role,
    }


def _fraction_value(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _diagnostics(
    *,
    response: Mapping[str, object],
    stage_kind: StageKind,
    request_fingerprint: str,
    model: ResolvedModel,
    attempt_count: int,
    validation_code: str | None,
    image_count: int,
    context_cue_count: int,
    duration_seconds: float,
) -> VisionInferenceDiagnostics:
    prompt_version, schema_version, stage_contract_version = _contract_versions(
        stage_kind
    )
    done_reason = response.get("done_reason")
    return VisionInferenceDiagnostics(
        request_fingerprint=request_fingerprint,
        model_name=model.configured_name,
        model_identity=model.execution_identity.identifier,
        runtime_identity=model.runtime_identity.identifier,
        prompt_version=prompt_version,
        schema_version=schema_version,
        stage_contract_version=stage_contract_version,
        retry_policy_version=RETRY_POLICY_VERSION,
        cache_hit=False,
        attempt_count=attempt_count,
        validation_code=validation_code,
        image_count=image_count,
        context_cue_count=context_cue_count,
        duration_seconds=duration_seconds,
        prompt_eval_count=_non_negative_int(response.get("prompt_eval_count")),
        eval_count=_non_negative_int(response.get("eval_count")),
        done_reason=(
            done_reason
            if isinstance(done_reason, str)
            and re.fullmatch(r"[0-9A-Za-z][0-9A-Za-z._:+/-]{0,255}", done_reason)
            else None
        ),
    )


def _contract_versions(stage_kind: StageKind) -> tuple[str, str, str]:
    if stage_kind == "scene_catalog":
        return (
            SCENE_CATALOG_PROMPT_VERSION,
            SCENE_CATALOG_SCHEMA_VERSION,
            SCENE_CATALOG_STAGE_CONTRACT_VERSION,
        )
    return (
        CANDIDATE_ANNOTATION_PROMPT_VERSION,
        CANDIDATE_ANNOTATION_SCHEMA_VERSION,
        CANDIDATE_ANNOTATION_STAGE_CONTRACT_VERSION,
    )


def _non_negative_int(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _http_failure(
    status_code: int,
    retry_after_seconds: float,
) -> VisionRuntimeError:
    if status_code in {408, 429} or status_code >= 500:
        return VisionRuntimeError(
            VisionRuntimeFailureReason.TRANSPORT_FAILURE,
            validation_code="ollama_transport_failure",
            retry_after_seconds=retry_after_seconds,
        )
    reason = (
        VisionRuntimeFailureReason.MODEL_UNAVAILABLE
        if status_code == 404
        else VisionRuntimeFailureReason.INVALID_REQUEST
    )
    return VisionRuntimeError(reason)


def _repair_validation_code(error: VisionRuntimeError) -> str | None:
    """model出力を検証できたfailureだけをprompt修復指示へ変換する。"""
    if (
        error.reason not in _PROMPT_REPAIR_REASONS
        or error.validation_code == "ollama_model_identity_response_invalid"
    ):
        return None
    return error.validation_code


def _require_model_role(
    model: ResolvedModel,
    expected_role: ModelRole,
    num_ctx: int,
) -> None:
    if (
        model.role is not expected_role
        or model.execution_identity.store_kind is not ModelStoreKind.OLLAMA
        or num_ctx < 1
    ):
        raise VisionRuntimeError(VisionRuntimeFailureReason.INVALID_REQUEST)
