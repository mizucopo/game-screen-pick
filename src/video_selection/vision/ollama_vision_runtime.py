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
    CONTEXT_CUE_RELEVANCES,
    EXPLANATION_VALUES,
    SCREEN_TEXT_KINDS,
    SPOILER_RISKS,
    CandidateAnnotation,
    ContextCueRelevance,
    ExplanationValue,
    ScreenTextKind,
    SpoilerRisk,
    candidate_annotation_context_is_valid,
    candidate_annotation_relationships_are_valid,
    privacy_safe_candidate_text,
)
from ..models.candidate_annotation_request import CandidateAnnotationRequest
from ..models.candidate_frame_observation import (
    CANDIDATE_FRAME_CONTENT_KINDS,
    CANDIDATE_INTERFACE_KINDS,
    CHARACTER_BODY_VISIBILITIES,
    DIALOGUE_TEXT_PRESENTATIONS,
    PRIMARY_SUBJECT_VISIBILITIES,
    TRANSIENT_OBSTRUCTIONS,
    CandidateFrameContentKind,
    CandidateFrameObservation,
    CandidateInterfaceKind,
    CharacterBodyVisibility,
    DialogueTextPresentation,
    PrimarySubjectVisibility,
    TransientObstruction,
)
from ..models.frame_candidate import FrameCandidate
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
from ..models.scene_kind import SCENE_KINDS, SceneKind
from ..models.vision_inference_diagnostics import VisionInferenceDiagnostics
from ..models.vision_runtime_error import VisionRuntimeError
from ..models.vision_runtime_failure_reason import VisionRuntimeFailureReason
from ..services.gpu_work_coordinator import GpuWorkCoordinator
from ..services.select_representative_candidate_frame_observation import (
    select_representative_candidate_frame_observation,
)
from ..utils.http_retry_delay import http_retry_delay
from .vision_contract import (
    CANDIDATE_ANNOTATION_PROMPT_VERSION,
    CANDIDATE_ANNOTATION_SCHEMA,
    CANDIDATE_ANNOTATION_SCHEMA_VERSION,
    CANDIDATE_ANNOTATION_STAGE_CONTRACT_VERSION,
    COMBAT_ENCOUNTER_CONFIRMATION_PROMPT_VERSION,
    COMBAT_ENCOUNTER_CONFIRMATION_STAGE_CONTRACT_VERSION,
    COMBAT_ENCOUNTER_VERIFICATION_PROMPT_VERSION,
    COMBAT_ENCOUNTER_VERIFICATION_SCHEMA,
    COMBAT_ENCOUNTER_VERIFICATION_SCHEMA_VERSION,
    COMBAT_ENCOUNTER_VERIFICATION_STAGE_CONTRACT_VERSION,
    COMBAT_VISIBILITY_CONFIRMATION_PROMPT_VERSION,
    COMBAT_VISIBILITY_CONFIRMATION_STAGE_CONTRACT_VERSION,
    COMBAT_VISIBILITY_EDGE_AUDIT_PROMPT_VERSION,
    COMBAT_VISIBILITY_EDGE_AUDIT_STAGE_CONTRACT_VERSION,
    COMBAT_VISIBILITY_VERIFICATION_PROMPT_VERSION,
    COMBAT_VISIBILITY_VERIFICATION_SCHEMA,
    COMBAT_VISIBILITY_VERIFICATION_SCHEMA_VERSION,
    COMBAT_VISIBILITY_VERIFICATION_STAGE_CONTRACT_VERSION,
    PUBLICATION_BOUNDARY_VERIFICATION_PROMPT_VERSION,
    PUBLICATION_BOUNDARY_VERIFICATION_SCHEMA,
    PUBLICATION_BOUNDARY_VERIFICATION_SCHEMA_VERSION,
    PUBLICATION_BOUNDARY_VERIFICATION_STAGE_CONTRACT_VERSION,
    RETRY_POLICY_VERSION,
    SCENE_CATALOG_PROMPT_VERSION,
    SCENE_CATALOG_SCHEMA,
    SCENE_CATALOG_SCHEMA_VERSION,
    SCENE_CATALOG_STAGE_CONTRACT_VERSION,
    VISION_GENERATION_SEED,
)

JsonRequester = Callable[
    [str, str, Mapping[str, object] | None, float],
    object,
]
Sleeper = Callable[[float], None]
ModelStateResolver = Callable[[ResolvedModel], ModelArtifact]
InferenceValue = TypeVar("InferenceValue")
InferenceParser = Callable[[Mapping[str, object]], InferenceValue]
StageKind = Literal[
    "scene_catalog",
    "candidate_annotation",
    "combat_encounter_confirmation",
    "combat_encounter_verification",
    "combat_visibility_confirmation",
    "combat_visibility_edge_audit",
    "combat_visibility_verification",
    "publication_boundary_verification",
]
OpponentBodyFraming = Literal["complete", "edge_cropped", "occluded", "absent"]


def _generation_options(num_ctx: int) -> dict[str, int]:
    """全Vision operationで共有する再現可能なOllama optionsを返す。"""
    return {
        "temperature": 0,
        "num_ctx": num_ctx,
        "seed": VISION_GENERATION_SEED,
    }


def _semantic_generation_options() -> dict[str, object]:
    """Stage fingerprintへ含める再現可能な生成条件を返す。"""
    return {
        "temperature": 0,
        "stream": False,
        "think": False,
        "seed": VISION_GENERATION_SEED,
    }


_SCENE_ENTRY_KEYS = {
    "slug",
    "display_name",
    "description",
    "scene_kind",
    "selection_role",
}
_ANNOTATION_KEYS = {
    "frame_observations",
    "context_relevance",
    "supporting_context_cue_ids",
}
_FRAME_OBSERVATION_KEYS = {
    "frame_id",
    "scene_slug",
    "content_kind",
    "interface_kind",
    "prominent_event_portrait",
    "cinematic_event_presentation",
    "on_screen_dialogue_text_visible",
    "dialogue_text_presentation",
    "visible_action",
    "visible_character_or_enemy",
    "combat_action",
    "player_body_visibility",
    "opponent_body_visibility",
    "effect_only_frame",
    "explanation_value",
    "screen_text_kind",
    "primary_subject_visibility",
    "transient_obstruction",
    "spoiler_risk",
    "spoiler_evidence",
}
_COMBAT_VISIBILITY_VERIFICATION_KEYS = {
    "effect_screen_coverage",
    "largest_foreground_element",
    "player_body_visibility",
    "opponent_body_visibility",
    "opponent_body_framing",
    "effect_overlaps_combatant_body",
    "effect_only_frame",
}
_COMBAT_ENCOUNTER_VERIFICATION_KEYS = {
    "combat_encounter_visible",
    "combat_encounter_evidence",
}
_COMBAT_ENCOUNTER_EVIDENCE = {
    "none",
    "enemy_status_ui",
    "opposing_bodies",
    "both",
}
_PUBLICATION_BOUNDARY_VERIFICATION_KEYS = {
    "transient_transition_effect",
    "transition_effect_kind",
    "transition_effect_coverage",
    "cinematic_letterbox",
    "event_staging",
    "on_screen_dialogue_text_visible",
    "visible_character_action",
    "primary_content_readability",
}
_EFFECT_SCREEN_COVERAGES = {
    "none",
    "under_quarter",
    "quarter_to_half",
    "over_half",
}
_LARGEST_FOREGROUND_ELEMENTS = {
    "player_body",
    "opponent_body",
    "other_character_body",
    "environment",
    "interface",
    "visual_effect",
    "unclear",
}
_EFFECT_COMBATANT_OVERLAPS = {"none", "partial", "severe"}
_OPPONENT_BODY_FRAMINGS: tuple[OpponentBodyFraming, ...] = (
    "complete",
    "edge_cropped",
    "occluded",
    "absent",
)
_TRANSITION_EFFECT_KINDS = {
    "none",
    "white_wipe",
    "motion_blur_or_streak",
    "fade",
    "other",
}
_PRIMARY_CONTENT_READABILITIES = {"clear", "partial", "obscured"}
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
    "scene_kindはcombat=敵またはbossとの戦闘、exploration=探索・移動・puzzle、"
    "interface=menu・map・shop・save・tutorial・document・title、"
    "event=会話・cutscene・物語event、other=どれにも該当しない場面です。"
    "slug=otherのscene_kindは必ずotherにします。"
    "scene_kindは複数sceneで重複して構いませんが、slugはcatalog内で一意にします。"
    "同じscene_kindのsceneを複数作る場合は、battle・boss-battle、shop・mapのように"
    "視覚的・説明上の役割を区別する一意なslugを付けます。"
    "selection_roleはordinary=通常の単発scene、cinematic=会話・演出・eventが主体、"
    "recurring_gameplay=戦闘UI・探索・puzzleなど繰り返し現れるplay構造です。"
    "同じ画面構造を一時的な敵やエフェクトだけで別sceneへ分割しません。"
    "sceneはブログで役割が異なる視覚・内容のまとまりとして作ります。\n"
)
_CANDIDATE_FRAME_DIRECT_OBSERVATION_INSTRUCTION = (
    "この画像だけに実際に見えるものを最初に観測してください。"
    "手紙・手記・日誌・記録を読む画面ならinterface_kind=documentです。"
    "prominent_event_portraitは会話やeventの演出として大きな人物立ち絵・胸像が"
    "gameplay画面へ重なる場合だけtrueです。画面隅の小さな円形・枠付きの常設HUD"
    "portraitはfalseです。"
    "cinematic_event_presentationは上下の映画的な黒帯、操作HUDのない固定camera、"
    "会話・event用に人物やNPCを並べた構図など、通常操作ではなくeventやcutsceneの"
    "提示だと画面自体から分かる場合だけtrueです。通常の戦闘・探索HUD、操作中の"
    "gameplay、画面隅の常設portraitはfalseです。"
    "画像内で実際の台詞文字が読める場合だけ"
    "on_screen_dialogue_text_visible=trueです。音声やContext Cueに会話文があっても、"
    "画像内で文字を読めなければfalseです。dialogue_text_presentationは画像内で"
    "読める台詞文字の表示形式をdialogue_box・speech_bubble・subtitle_overlay・other"
    "から選び、音声やContext Cueしかない場合はnoneです。人物portraitだけ、空の"
    "台詞欄、見出し、説明、目的表示、操作案内、item名、HUDは台詞ではありません。"
    "人物または敵の具体的な動作や相互作用がなければvisible_action=falseです。"
    "静止した立ち姿、空の背景、建物、移動先表示は動作ではありません。"
    "人物・NPC・player・monster・bossの本体を判別できなければ"
    "visible_character_or_enemy=falseです。portrait、HUD、文字、影、発光、"
    "移動軌跡だけは本体ではありません。戦闘ではplayer本体と攻撃相手本体を別々に"
    "判定し、portrait、HUD、文字、光、hit effect、影を本体に数えません。"
)
_COMBAT_ENCOUNTER_VERIFICATION_INSTRUCTION = (
    "この画像1枚に実際に見える画素だけを観測してください。音声、前後場面、"
    "説明文は使いません。combat_encounter_visibleは、敵またはboss固有の名前と"
    "HP・status barがある、またはplayer本体と攻撃相手本体が戦闘中だと画面から"
    "分かる場合にtrueです。敵本体が画面端で切れる、エフェクトに隠れる、画面外に"
    "いる場合でも、敵・boss固有の名前とHP・status barがあればtrueです。player自身の"
    "通常HP、portrait、操作button、minimapだけではfalseです。"
    "combat_encounter_evidenceはnone、enemy_status_ui、opposing_bodies、bothから"
    "選びます。combat_encounter_visibleがfalseならnone、trueならnone以外です。"
)
_COMBAT_VISIBILITY_VERIFICATION_INSTRUCTION = (
    "この画像1枚に実際に見える画素だけを観測してください。音声、前後場面、"
    "説明文は使いません。visual_effectは攻撃の光、爆発、煙、軌跡、白飛びなど"
    "一時的な演出です。effect_screen_coverageはvisual_effectが画面全体に占める"
    "面積をnone、under_quarter、quarter_to_half、over_halfから選びます。"
    "largest_foreground_elementは前景で最も大きく目立つものをplayer_body、"
    "opponent_body、other_character_body、environment、interface、visual_effect、"
    "unclearから選びます。player_body_visibilityとopponent_body_visibilityは、"
    "本体の頭部または上端から足元または下端までの輪郭と姿勢が画面内で明瞭なら"
    "clear、本体は判別できても画像の端で大きく切れるかエフェクト等に隠れるなら"
    "partial、本体を判別できなければabsentです。光、影、名前、HUDを本体に"
    "数えません。opponent_body_framingは攻撃相手本体の主要部が画面内に収まり"
    "輪郭と姿勢を追えるならcomplete、本体が画像の端で大きく切れるなら"
    "edge_cropped、画像内にはあるがエフェクト等で大きく隠れるならoccluded、"
    "本体を判別できなければabsentです。effect_overlaps_combatant_bodyは"
    "visual_effectがplayerまたはopponentの本体へ重なる程度をnone、partial、"
    "severeから選びます。"
    "effect_only_frameは画面中央の主内容が一時的な光・爆発・煙だけで、人物・敵・"
    "物体の本体を主対象として一つも明瞭に判別できない場合だけtrueです。敵や人物"
    "の本体が一体でもclearならfalseです。"
)
_INDEPENDENT_CONFIRMATION_INSTRUCTION = (
    "これは掲載可否を確定する独立した再確認です。先の回答を推測せず、"
    "画像の画素を最初から観測し直してください。"
)
_COMBAT_VISIBILITY_EDGE_AUDIT_INSTRUCTION = (
    "掲載可能と判断する前に、画像の上端、下端、左端、右端を順に確認してください。"
    "攻撃相手本体の主要な輪郭がどれかの画像端で切れている場合は、"
    "opponent_body_framingを必ずedge_croppedにし、opponent_body_visibilityを"
    "clearにしません。敵名、HP bar、光、攻撃effectを敵本体と取り違えません。"
)
_PUBLICATION_BOUNDARY_VERIFICATION_INSTRUCTION = (
    "この画像1枚に実際に見える画素だけを観測してください。音声、前後場面、"
    "説明文は使いません。transient_transition_effectは、画面の切替や移動中だけ"
    "現れる白いwipe、太い光帯、fade、motion blurやstreakが、安定した画面内容を"
    "横切り隠している場合だけtrueです。地図の雲、通常のcursor、選択marker、"
    "常設UIはfalseです。transition_effect_kindはnone、white_wipe、"
    "motion_blur_or_streak、fade、otherから選びます。transition_effect_coverageは"
    "一時的な切替effectが画面に占める面積をnone、under_quarter、quarter_to_half、"
    "over_halfから選びます。cinematic_letterboxは画面上端と下端の両方に太い"
    "黒帯がある場合だけtrueです。event_stagingは複数人物が会話やevent用の固定構図で"
    "向き合う、並ぶ、囲む場合だけtrueです。通常操作中に偶然近くにいるだけなら"
    "falseです。on_screen_dialogue_text_visibleは登場人物の台詞文字が画像内で"
    "実際に読める場合だけtrueです。目的表示、地名、menu、操作案内は台詞では"
    "ありません。visible_character_actionは人物同士の具体的な動作や相互作用が"
    "画像内で明瞭な場合だけtrueです。静止して立つ、向き合う、並ぶだけならfalseです。"
    "primary_content_readabilityは主内容が遮られず明瞭ならclear、一部隠れるなら"
    "partial、切替effect等で何の画面か分からないならobscuredです。"
)
_CANDIDATE_ANNOTATION_SEMANTICS = (
    "各frameを他のframeの内容と混ぜず、対応するframe_idごとに個別評価します。"
    "最初に各画像の直接観測を推測せず決め、その後で画面内容と説明価値を決めます。"
    "interface_kindは画面全体の主用途をnone・document・shop・map・save・"
    "tutorial_help・other_interface・titleから選びます。documentは手紙・手記・"
    "日誌・記録を読む画面です。戦闘HUDだけをother_interfaceにせず、"
    "戦闘や探索が主ならnoneにします。on_screen_dialogue_text_visibleは登場人物の"
    "台詞文字を画像内で実際に読めるときだけtrueです。音声やContext Cueに会話文が"
    "あっても、画像内で文字を読めなければfalseです。dialogue_text_presentationは"
    "画像内で読める台詞文字の表示形式をdialogue_box・speech_bubble・"
    "subtitle_overlay・otherから選び、音声やContext Cueしかない場合はnoneです。"
    "人物portraitだけ、空の台詞欄、見出し、説明、目的表示、操作案内、item名、HUD"
    "ならfalseです。visible_actionは人物または"
    "敵の具体的な動作や相互作用が見えるときだけtrueで、静止した立ち姿、空の背景、"
    "建物、移動先表示だけならfalseです。visible_character_or_enemyは人物・NPC・player・"
    "monster・bossの本体を判別できるときだけtrueで、portrait、HUD、文字、影、発光、"
    "移動軌跡だけは数えません。prominent_event_portraitは会話やeventの演出として"
    "大きな人物立ち絵・胸像がgameplay画面へ重なる場合だけtrueです。画面隅の小さな"
    "円形・枠付きの常設HUD portraitはfalseです。\n"
    "cinematic_event_presentationは上下の映画的な黒帯、操作HUDのない固定camera、"
    "会話・event用に人物やNPCを並べた構図など、通常操作ではなくeventやcutsceneの"
    "提示だと画面自体から分かる場合だけtrueです。通常の戦闘・探索HUD、操作中の"
    "gameplay、画面隅の常設portraitはfalseです。\n"
    "combat_actionはplayerと敵が戦っている場面だけtrueです。"
    "player_body_visibilityとopponent_body_visibilityは、操作するplayer本体と攻撃する"
    "相手本体の輪郭・姿勢が明瞭ならclear、一部が隠れるならpartial、本体を判別"
    "できなければabsentです。portrait、HUD、文字、光、hit effect、影を本体に"
    "数えません。effect_only_frameは画面中央の主内容が一時的な光・爆発・煙だけで、"
    "人物・敵・物体の本体を主対象として一つも明瞭に判別できない場合だけtrueです。"
    "敵や人物の本体が一体でもclearならfalseです。\n"
    "gameplay_action=操作・戦闘・探索の具体的な動作、gameplay_idle=人物や背景が"
    "見えても具体的な動作がない通常画面、event_dialogue=frame内に台詞表示が"
    "実在する会話、event_action=台詞がなくても具体的な演出や動作が見える出来事、"
    "event_setup=出来事の開始待ちで動作も台詞表示もない画面、document=手紙・手記・"
    "日誌・記録を読む画面、shop・map・save・tutorial_help・other_interface="
    "各interface、title・other=その他の役割です。\n"
    "primary_subject_visibilityは人物・敵・品物・行動などブログ説明の主対象が"
    "clear・partial・absentのどれか、transient_obstructionは発光・白飛び・移動・"
    "画面切替による一時的な遮蔽がnone・partial・severeのどれかを返します。"
    "大きな発光やエフェクトで主対象が隠れるframe、白飛び、移動・画面切替の"
    "途中はsevereにします。\n"
    "explanation_valueのnone=主対象や出来事を説明できずブログ掲載価値がない、"
    "low=判別できるが汎用的・重複的、medium=具体的なplay状態や出来事を説明できる、"
    "high=重要な主対象・行動・関係が明瞭で本文を直接補強する、です。\n"
    "document、tutorial_help、主対象がabsent、event_setup、severeな遮蔽、"
    "effect_only_frameは"
    "explanation_valueをnoneにします。screen_text_kindはそのframe内に実際に"
    "見える文字の役割だけで決め、別frameやContext Cueから推測しません。\n"
    "context_relevanceのnone=近接していても画像説明と無関係、weak=補足になる、"
    "context_relevanceのstrong=画像の意味を特定するため不可欠、です。"
    "単にContext Cueが存在するだけで"
    "strongにしません。\n"
    "spoiler_riskはnone=汎用的な探索・戦闘、low=軽微な進行情報、medium=固有boss・"
    "終盤固有area・重要quest結果、spoiler_riskのhigh=ending・最終bossの正体や形態・"
    "主要人物の生死・裏切り・犯人や真の正体・中心的な種明かしです。"
    "進行位置だけではriskを上げません。\n"
)
_CONTENT_KIND_LABELS: Mapping[CandidateFrameContentKind, str] = {
    "gameplay_action": "具体的なプレイ",
    "gameplay_idle": "通常プレイの待機場面",
    "event_dialogue": "台詞のあるイベント",
    "event_action": "動きのあるイベント",
    "event_setup": "イベント開始前の場面",
    "document": "文書画面",
    "shop": "ショップ画面",
    "map": "マップ画面",
    "save": "セーブ画面",
    "tutorial_help": "チュートリアル画面",
    "other_interface": "操作画面",
    "title": "タイトル画面",
    "other": "その他の場面",
}


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
        catalog_response_count = 0

        def parse_catalog(value: Mapping[str, object]) -> SceneCatalog:
            nonlocal catalog_response_count
            catalog_response_count += 1
            return _parse_scene_catalog(
                value,
                repair_duplicate_slugs=catalog_response_count > 1,
            )

        return self._infer(
            stage_kind="scene_catalog",
            request_fingerprint=_fingerprint(semantic_input),
            payload=_scene_catalog_payload(request, model, num_ctx),
            parser=parse_catalog,
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
        candidate_response_count = 0

        def parse_candidate(
            value: Mapping[str, object],
        ) -> tuple[CandidateAnnotation, bool, bool, bool, bool]:
            nonlocal candidate_response_count
            candidate_response_count += 1
            (
                annotation,
                redacted,
                requires_dialogue_verification,
                requires_combat_verification,
                requires_combat_encounter_verification,
                requires_publication_verification,
            ) = _parse_candidate_annotation(value, request, catalog)
            if requires_dialogue_verification and candidate_response_count == 1:
                raise _domain_error(
                    "candidate_annotation_dialogue_visibility_unverified"
                )
            return (
                annotation,
                redacted,
                requires_combat_verification,
                requires_combat_encounter_verification,
                requires_publication_verification,
            )

        (
            (
                annotation,
                free_text_redacted,
                requires_combat_verification,
                requires_combat_encounter_verification,
                requires_publication_verification,
            ),
            diagnostics,
        ) = self._infer(
            stage_kind="candidate_annotation",
            request_fingerprint=_fingerprint(semantic_input),
            payload=_candidate_payload(request, catalog, model, num_ctx),
            parser=parse_candidate,
            model=model,
            image_count=len(request.frame_candidates),
            context_cue_count=len(request.context_cues),
        )
        combat_scene = catalog.for_slug(annotation.scene_slug).scene_kind == "combat"
        if requires_combat_encounter_verification:
            verification_input = _combat_encounter_verification_semantic_input(
                annotation.candidate,
                model,
                num_ctx,
            )
            combat_encounter_visible, verification_diagnostics = self._infer(
                stage_kind="combat_encounter_verification",
                request_fingerprint=_fingerprint(verification_input),
                payload=_combat_encounter_verification_payload(
                    annotation.candidate,
                    model,
                    num_ctx,
                ),
                parser=_parse_combat_encounter_verification,
                model=model,
                image_count=1,
                context_cue_count=0,
            )
            diagnostics = _merge_candidate_diagnostics(
                diagnostics,
                verification_diagnostics,
            )
            if not combat_encounter_visible:
                confirmation_input = _combat_encounter_verification_semantic_input(
                    annotation.candidate,
                    model,
                    num_ctx,
                    independently_confirm=True,
                )
                combat_encounter_visible, confirmation_diagnostics = self._infer(
                    stage_kind="combat_encounter_confirmation",
                    request_fingerprint=_fingerprint(confirmation_input),
                    payload=_combat_encounter_verification_payload(
                        annotation.candidate,
                        model,
                        num_ctx,
                        independently_confirm=True,
                    ),
                    parser=_parse_combat_encounter_verification,
                    model=model,
                    image_count=1,
                    context_cue_count=0,
                )
                diagnostics = _merge_candidate_diagnostics(
                    diagnostics,
                    confirmation_diagnostics,
                )
            if not combat_encounter_visible and combat_scene:
                annotation = replace(annotation, explanation_value="none")
            requires_combat_verification = combat_encounter_visible
        requires_noncombat_visibility_verification = (
            requires_combat_encounter_verification
            and not requires_combat_verification
            and annotation.explanation_value != "none"
        )
        if requires_combat_verification or requires_noncombat_visibility_verification:
            verification_input = _combat_visibility_verification_semantic_input(
                annotation.candidate,
                model,
                num_ctx,
            )
            combat_visibility, verification_diagnostics = self._infer(
                stage_kind="combat_visibility_verification",
                request_fingerprint=_fingerprint(verification_input),
                payload=_combat_visibility_verification_payload(
                    annotation.candidate,
                    model,
                    num_ctx,
                ),
                parser=_parse_combat_visibility_verification,
                model=model,
                image_count=1,
                context_cue_count=0,
            )
            diagnostics = _merge_candidate_diagnostics(
                diagnostics,
                verification_diagnostics,
            )
            first_combat_visibility = combat_visibility
            confirmed_combat_visibility = None
            if (
                _is_publishable_combat_visibility(first_combat_visibility)
                or requires_noncombat_visibility_verification
            ):
                confirmation_input = _combat_visibility_verification_semantic_input(
                    annotation.candidate,
                    model,
                    num_ctx,
                    independently_confirm=True,
                )
                combat_visibility, confirmation_diagnostics = self._infer(
                    stage_kind="combat_visibility_confirmation",
                    request_fingerprint=_fingerprint(confirmation_input),
                    payload=_combat_visibility_verification_payload(
                        annotation.candidate,
                        model,
                        num_ctx,
                        independently_confirm=True,
                    ),
                    parser=_parse_combat_visibility_verification,
                    model=model,
                    image_count=1,
                    context_cue_count=0,
                )
                diagnostics = _merge_candidate_diagnostics(
                    diagnostics,
                    confirmation_diagnostics,
                )
                confirmed_combat_visibility = combat_visibility
            combat_is_consistently_publishable = (
                confirmed_combat_visibility is not None
                and _is_publishable_combat_visibility(first_combat_visibility)
                and _is_publishable_combat_visibility(confirmed_combat_visibility)
            )
            if requires_noncombat_visibility_verification:
                visibility_is_acceptable = (
                    confirmed_combat_visibility is not None
                    and _is_consistent_noncombat_or_publishable_combat_visibility(
                        first_combat_visibility,
                        confirmed_combat_visibility,
                    )
                )
            else:
                visibility_is_acceptable = combat_is_consistently_publishable
            if visibility_is_acceptable and combat_is_consistently_publishable:
                edge_audit_input = _combat_visibility_edge_audit_semantic_input(
                    annotation.candidate,
                    model,
                    num_ctx,
                )
                edge_audit, edge_audit_diagnostics = self._infer(
                    stage_kind="combat_visibility_edge_audit",
                    request_fingerprint=_fingerprint(edge_audit_input),
                    payload=_combat_visibility_edge_audit_payload(
                        annotation.candidate,
                        model,
                        num_ctx,
                    ),
                    parser=_parse_combat_visibility_verification,
                    model=model,
                    image_count=1,
                    context_cue_count=0,
                )
                diagnostics = _merge_candidate_diagnostics(
                    diagnostics,
                    edge_audit_diagnostics,
                )
                visibility_is_acceptable = _is_publishable_combat_visibility(edge_audit)
            if not visibility_is_acceptable:
                annotation = replace(annotation, explanation_value="none")
        elif requires_publication_verification:
            verification_input = _publication_boundary_verification_semantic_input(
                annotation.candidate,
                model,
                num_ctx,
            )
            (
                (
                    transient_transition_effect,
                    cinematic_letterbox,
                    event_staging,
                    on_screen_dialogue_text_visible,
                    visible_character_action,
                ),
                verification_diagnostics,
            ) = self._infer(
                stage_kind="publication_boundary_verification",
                request_fingerprint=_fingerprint(verification_input),
                payload=_publication_boundary_verification_payload(
                    annotation.candidate,
                    model,
                    num_ctx,
                ),
                parser=_parse_publication_boundary_verification,
                model=model,
                image_count=1,
                context_cue_count=0,
            )
            static_event_setup = (
                cinematic_letterbox
                and event_staging
                and not on_screen_dialogue_text_visible
                and not visible_character_action
            )
            if transient_transition_effect or static_event_setup:
                annotation = replace(annotation, explanation_value="none")
            diagnostics = _merge_candidate_diagnostics(
                diagnostics,
                verification_diagnostics,
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
        "options": _generation_options(num_ctx),
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
        "入力された1〜3枚を個別に評価し、共有Scene Catalogを使って"
        "frame_observationsを返してください。"
        + _CANDIDATE_ANNOTATION_SEMANTICS
        + "frame_observationsは全frame_candidate_idsを入力順に一度ずつ含め、"
        "frame_idは対応する個別画像のID、scene_slugはscene_catalogから選びます。"
        "画像品質、confidence、final score、eligible、selected、逐語的画面文、"
        "推論過程は出力しません。Context Cue本文をspoiler_evidenceへ引用しません。"
        "正規化後3〜5文字のCueは全文、6文字以上のCueは6文字以上の連続部分も"
        "spoiler_evidenceへ再出力しません。context_cuesが空ならcontext_relevanceは"
        "unavailable、supporting_context_cue_idsは空配列にします。context_cuesが"
        "ある場合はcontext_relevanceをunavailableにせず、weakまたはstrongなら"
        "supporting_context_cue_idsへ入力内のIDを1件以上入れ、noneなら空配列に"
        "します。各frameのspoiler_riskがnoneならspoiler_evidenceは空文字列にし、"
        "それ以外ならそのframeから判断できる根拠を空でない自分の言葉で記述します。\n"
        + json.dumps(semantic_request, ensure_ascii=False, sort_keys=True)
    )
    frame_messages = [
        {
            "role": "user",
            "content": (
                f"frame_candidate_id={item.identifier}。"
                f"{_CANDIDATE_FRAME_DIRECT_OBSERVATION_INSTRUCTION}"
            ),
            "images": [base64.b64encode(item.image_bytes).decode()],
        }
        for item in request.frame_candidates
    ]
    return {
        "model": model.configured_name,
        "stream": False,
        "think": False,
        "format": _candidate_schema(request, catalog),
        "options": _generation_options(num_ctx),
        "messages": [*frame_messages, {"role": "user", "content": content}],
    }


def _combat_encounter_verification_payload(
    candidate: FrameCandidate,
    model: ResolvedModel,
    num_ctx: int,
    *,
    independently_confirm: bool = False,
) -> dict[str, object]:
    """曖昧なactionが戦闘かを一画像へ確認するrequestを返す。"""
    return {
        "model": model.configured_name,
        "stream": False,
        "think": False,
        "format": COMBAT_ENCOUNTER_VERIFICATION_SCHEMA,
        "options": _generation_options(num_ctx),
        "messages": [
            {
                "role": "user",
                "content": _COMBAT_ENCOUNTER_VERIFICATION_INSTRUCTION
                + (
                    _INDEPENDENT_CONFIRMATION_INSTRUCTION
                    if independently_confirm
                    else ""
                ),
                "images": [base64.b64encode(candidate.image_bytes).decode()],
            }
        ],
    }


def _combat_visibility_verification_payload(
    candidate: FrameCandidate,
    model: ResolvedModel,
    num_ctx: int,
    *,
    independently_confirm: bool = False,
) -> dict[str, object]:
    """戦闘の掲載境界だけを一画像へ確認する小さなstructured requestを返す。"""
    return {
        "model": model.configured_name,
        "stream": False,
        "think": False,
        "format": COMBAT_VISIBILITY_VERIFICATION_SCHEMA,
        "options": _generation_options(num_ctx),
        "messages": [
            {
                "role": "user",
                "content": _COMBAT_VISIBILITY_VERIFICATION_INSTRUCTION
                + (
                    _INDEPENDENT_CONFIRMATION_INSTRUCTION
                    if independently_confirm
                    else ""
                ),
                "images": [base64.b64encode(candidate.image_bytes).decode()],
            }
        ],
    }


def _combat_visibility_edge_audit_payload(
    candidate: FrameCandidate,
    model: ResolvedModel,
    num_ctx: int,
) -> dict[str, object]:
    """敵本体が画像端で欠けるfalse positiveを監査するrequestを返す。"""
    return {
        "model": model.configured_name,
        "stream": False,
        "think": False,
        "format": COMBAT_VISIBILITY_VERIFICATION_SCHEMA,
        "options": _generation_options(num_ctx),
        "messages": [
            {
                "role": "user",
                "content": (
                    _COMBAT_VISIBILITY_VERIFICATION_INSTRUCTION
                    + _INDEPENDENT_CONFIRMATION_INSTRUCTION
                    + _COMBAT_VISIBILITY_EDGE_AUDIT_INSTRUCTION
                ),
                "images": [base64.b64encode(candidate.image_bytes).decode()],
            }
        ],
    }


def _publication_boundary_verification_payload(
    candidate: FrameCandidate,
    model: ResolvedModel,
    num_ctx: int,
) -> dict[str, object]:
    """遷移と静止eventの掲載境界を一画像へ確認するrequestを返す。"""
    return {
        "model": model.configured_name,
        "stream": False,
        "think": False,
        "format": PUBLICATION_BOUNDARY_VERIFICATION_SCHEMA,
        "options": _generation_options(num_ctx),
        "messages": [
            {
                "role": "user",
                "content": _PUBLICATION_BOUNDARY_VERIFICATION_INSTRUCTION,
                "images": [base64.b64encode(candidate.image_bytes).decode()],
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
    observation_array = properties["frame_observations"]
    observation_array["minItems"] = len(request.frame_candidates)
    observation_array["maxItems"] = len(request.frame_candidates)
    observation_items = cast(dict[str, object], observation_array["items"])
    observation_properties = cast(
        dict[str, dict[str, object]],
        observation_items["properties"],
    )
    observation_properties["frame_id"]["enum"] = [
        item.identifier for item in request.frame_candidates
    ]
    observation_properties["scene_slug"]["enum"] = list(catalog.slugs)
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
    content = cast(str, messages[-1]["content"])
    repair = f"前回の出力を修正してください。validation_code={validation_code}"
    recheck_candidate_observations = validation_code.startswith("candidate_annotation_")
    if validation_code == "candidate_annotation_relationship_invalid":
        repair += (
            "\n関係を必ず修正します。spoiler_riskがnoneならspoiler_evidenceは"
            "空文字列、low・medium・highならspoiler_evidenceは画面から判断した"
            "根拠を1文以上記述します。context_relevanceがnoneまたはunavailable"
            "ならsupporting_context_cue_idsは空配列、weakまたはstrongなら入力内IDを"
            "1件以上入れます。"
        )
    if validation_code == "scene_catalog_domain_invalid":
        repair += (
            "\nscene slugをcatalog内で重複させません。scene_kindは重複可能ですが、"
            "同じscene_kindの各sceneには視覚的・説明上の役割を区別する一意なslugを"
            "付けます。slug=otherは一件だけにし、そのscene_kindはother、"
            "selection_roleはordinaryにします。"
        )
    if recheck_candidate_observations:
        repair += (
            "\n画面内台詞文字を画像だけに対して再確認します。音声やContext Cueを"
            "根拠にしません。画像内で台詞文字を実際に読める場合だけ"
            "on_screen_dialogue_text_visible=trueとし、対応する"
            "dialogue_text_presentationを返します。人物portraitや会話中らしい構図"
            "だけで文字を読めない場合はfalseかつnoneにします。"
        )
    if recheck_candidate_observations:
        repair += (
            "\n再確認時もspoiler_riskがnoneならspoiler_evidenceは空文字列、"
            "low・medium・highなら画面から判断した根拠を1文以上記述します。"
            "context_relevanceがnoneまたはunavailableならsupporting_context_cue_idsは"
            "空配列、weakまたはstrongなら入力内IDを1件以上入れます。"
        )
    messages[-1]["content"] = f"{content}\n{repair}"
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


def _parse_scene_catalog(
    value: Mapping[str, object],
    *,
    repair_duplicate_slugs: bool = False,
) -> SceneCatalog:
    scenes = value.get("scenes")
    if (
        set(value) != {"scenes"}
        or not isinstance(scenes, list)
        or not 3 <= len(scenes) <= 8
    ):
        raise _schema_error("scene_catalog_schema_invalid")
    entries: list[SceneCatalogEntry] = []
    used_slugs: set[str] = set()
    for raw_scene in scenes:
        if not isinstance(raw_scene, dict) or set(raw_scene) != _SCENE_ENTRY_KEYS:
            raise _schema_error("scene_catalog_schema_invalid")
        slug = raw_scene.get("slug")
        display_name = raw_scene.get("display_name")
        description = raw_scene.get("description")
        scene_kind = raw_scene.get("scene_kind")
        selection_role = raw_scene.get("selection_role")
        if (
            not isinstance(slug, str)
            or not is_valid_scene_slug(slug)
            or not isinstance(display_name, str)
            or not display_name.strip()
            or not isinstance(description, str)
            or not description.strip()
            or scene_kind not in SCENE_KINDS
            or selection_role not in SCENE_SELECTION_ROLES
        ):
            raise _schema_error("scene_catalog_schema_invalid")
        if scene_kind == "other":
            slug = "other"
        if slug in used_slugs:
            if not repair_duplicate_slugs or slug == "other":
                raise _domain_error("scene_catalog_domain_invalid")
            slug = _unique_scene_slug(slug, used_slugs)
        used_slugs.add(slug)
        entries.append(
            SceneCatalogEntry(
                slug=slug,
                display_name=display_name,
                description=description,
                scene_kind=cast(SceneKind, scene_kind),
                selection_role=cast(SceneSelectionRole, selection_role),
            )
        )
    try:
        return SceneCatalog(tuple(entries))
    except ValueError:
        raise _domain_error("scene_catalog_domain_invalid") from None


def _unique_scene_slug(slug: str, used_slugs: set[str]) -> str:
    """重複した非other slugへ入力順の決定的suffixを付ける。"""
    suffix = 2
    while f"{slug}-{suffix}" in used_slugs:
        suffix += 1
    return f"{slug}-{suffix}"


def _parse_candidate_annotation(
    value: Mapping[str, object],
    request: CandidateAnnotationRequest,
    catalog: SceneCatalog,
) -> tuple[CandidateAnnotation, bool, bool, bool, bool, bool]:
    if set(value) != _ANNOTATION_KEYS:
        raise _schema_error("candidate_annotation_schema_invalid")
    raw_observations = value.get("frame_observations")
    context_relevance = value.get("context_relevance")
    cue_ids = value.get("supporting_context_cue_ids")
    if (
        not isinstance(raw_observations, list)
        or len(raw_observations) != len(request.frame_candidates)
        or context_relevance not in CONTEXT_CUE_RELEVANCES
        or not isinstance(cue_ids, list)
        or not all(isinstance(item, str) for item in cue_ids)
        or len(cue_ids) != len(set(cue_ids))
    ):
        raise _schema_error("candidate_annotation_schema_invalid")
    frames = {item.identifier: item for item in request.frame_candidates}
    typed_context_relevance = context_relevance
    typed_cue_ids = tuple(cast(list[str], cue_ids))
    available_cue_ids = tuple(item.identifier for item in request.context_cues)
    if not candidate_annotation_context_is_valid(
        typed_context_relevance,
        typed_cue_ids,
        available_cue_ids,
    ):
        raise _domain_error("candidate_annotation_context_invalid")
    observations = _parse_candidate_frame_observations(
        raw_observations,
        frames=frames,
        catalog=catalog,
        context_relevance=typed_context_relevance,
        cue_ids=typed_cue_ids,
    )
    expected_frame_ids = tuple(item.identifier for item in request.frame_candidates)
    actual_frame_ids = tuple(item.candidate.identifier for item in observations)
    if actual_frame_ids != expected_frame_ids:
        raise _domain_error("candidate_annotation_frame_observations_mismatch")
    requires_dialogue_verification = bool(request.context_cues) and any(
        (
            observation.prominent_event_portrait
            or observation.cinematic_event_presentation
        )
        and observation.visible_dialogue_text
        for observation in observations
    )
    try:
        selected = select_representative_candidate_frame_observation(observations)
        requires_combat_verification = (
            selected.combat_action
            and selected.opponent_body_visibility == "clear"
            and not selected.effect_only_frame
            and selected.effective_explanation_value != "none"
        )
        scene = catalog.for_slug(selected.scene_slug)
        requires_combat_encounter_verification = (
            selected.effective_explanation_value != "none"
            and not selected.combat_action
            and (
                (
                    scene.scene_kind == "combat"
                    and selected.effective_content_kind
                    in {"gameplay_action", "gameplay_idle"}
                )
                or (
                    scene.selection_role == "recurring_gameplay"
                    and selected.effective_content_kind
                    in {"gameplay_action", "event_action"}
                )
            )
        )
        requires_publication_verification = (
            selected.effective_explanation_value != "none"
            and not requires_combat_verification
            and not requires_combat_encounter_verification
            and (
                selected.effective_content_kind == "map"
                or selected.interface_kind == "map"
                or scene.selection_role == "cinematic"
            )
        )
        content_label = _CONTENT_KIND_LABELS[selected.effective_content_kind]
        annotation_summary = f"{scene.display_name}の{content_label}"
        frame_choice_reason = f"{content_label}が候補内で最も明瞭なフレーム"
        (
            annotation_summary,
            frame_choice_reason,
            spoiler_evidence,
            free_text_redacted,
        ) = _privacy_safe_candidate_texts(
            annotation_summary=annotation_summary,
            frame_choice_reason=frame_choice_reason,
            spoiler_evidence=selected.spoiler_evidence,
            scene_slug=selected.scene_slug,
            blog_image_type=selected.blog_image_type,
            spoiler_risk=selected.spoiler_risk,
            raw_context_texts=tuple(item.text for item in request.context_cues),
            catalog=catalog,
        )
        return (
            CandidateAnnotation(
                candidate=selected.candidate,
                summary=annotation_summary,
                candidate_moment_id=request.moment.identifier,
                scene_slug=selected.scene_slug,
                blog_image_type=selected.blog_image_type,
                explanation_value=selected.effective_explanation_value,
                frame_choice_reason=frame_choice_reason,
                screen_text_kind=selected.effective_screen_text_kind,
                context_relevance=typed_context_relevance,
                supporting_context_cue_ids=typed_cue_ids,
                spoiler_risk=selected.spoiler_risk,
                spoiler_evidence=spoiler_evidence,
            ),
            free_text_redacted,
            requires_dialogue_verification,
            requires_combat_verification,
            requires_combat_encounter_verification,
            requires_publication_verification,
        )
    except ValueError:
        raise _domain_error("candidate_annotation_domain_invalid") from None


def _parse_combat_encounter_verification(value: Mapping[str, object]) -> bool:
    """戦闘有無と根拠enumの関係を検証して戦闘有無だけを返す。"""
    if set(value) != _COMBAT_ENCOUNTER_VERIFICATION_KEYS:
        raise _schema_error("combat_encounter_verification_schema_invalid")
    combat_encounter_visible = value.get("combat_encounter_visible")
    combat_encounter_evidence = value.get("combat_encounter_evidence")
    if (
        not isinstance(combat_encounter_visible, bool)
        or combat_encounter_evidence not in _COMBAT_ENCOUNTER_EVIDENCE
        or combat_encounter_visible != (combat_encounter_evidence != "none")
    ):
        raise _schema_error("combat_encounter_verification_schema_invalid")
    return combat_encounter_visible


def _parse_combat_visibility_verification(
    value: Mapping[str, object],
) -> tuple[CharacterBodyVisibility, OpponentBodyFraming, bool]:
    """専用schemaの全fieldを検証し、掲載境界に必要な観測だけを返す。"""
    if set(value) != _COMBAT_VISIBILITY_VERIFICATION_KEYS:
        raise _schema_error("combat_visibility_verification_schema_invalid")
    effect_screen_coverage = value.get("effect_screen_coverage")
    largest_foreground_element = value.get("largest_foreground_element")
    player_body_visibility = value.get("player_body_visibility")
    opponent_body_visibility = value.get("opponent_body_visibility")
    opponent_body_framing = value.get("opponent_body_framing")
    effect_overlap = value.get("effect_overlaps_combatant_body")
    effect_only_frame = value.get("effect_only_frame")
    if (
        effect_screen_coverage not in _EFFECT_SCREEN_COVERAGES
        or largest_foreground_element not in _LARGEST_FOREGROUND_ELEMENTS
        or player_body_visibility not in CHARACTER_BODY_VISIBILITIES
        or opponent_body_visibility not in CHARACTER_BODY_VISIBILITIES
        or opponent_body_framing not in _OPPONENT_BODY_FRAMINGS
        or effect_overlap not in _EFFECT_COMBATANT_OVERLAPS
        or not isinstance(effect_only_frame, bool)
    ):
        raise _schema_error("combat_visibility_verification_schema_invalid")
    return opponent_body_visibility, opponent_body_framing, effect_only_frame


def _parse_publication_boundary_verification(
    value: Mapping[str, object],
) -> tuple[bool, bool, bool, bool, bool]:
    """遷移と静止eventの専用schemaを検証し掲載境界の観測を返す。"""
    if set(value) != _PUBLICATION_BOUNDARY_VERIFICATION_KEYS:
        raise _schema_error("publication_boundary_verification_schema_invalid")
    transient_transition_effect = value.get("transient_transition_effect")
    transition_effect_kind = value.get("transition_effect_kind")
    transition_effect_coverage = value.get("transition_effect_coverage")
    cinematic_letterbox = value.get("cinematic_letterbox")
    event_staging = value.get("event_staging")
    on_screen_dialogue_text_visible = value.get("on_screen_dialogue_text_visible")
    visible_character_action = value.get("visible_character_action")
    primary_content_readability = value.get("primary_content_readability")
    if (
        not isinstance(transient_transition_effect, bool)
        or not isinstance(cinematic_letterbox, bool)
        or not isinstance(event_staging, bool)
        or not isinstance(on_screen_dialogue_text_visible, bool)
        or not isinstance(visible_character_action, bool)
    ):
        raise _schema_error("publication_boundary_verification_schema_invalid")
    transition_relationship_is_valid = (
        transient_transition_effect
        and transition_effect_kind != "none"
        and transition_effect_coverage != "none"
    ) or (
        not transient_transition_effect
        and transition_effect_kind == "none"
        and transition_effect_coverage == "none"
    )
    if (
        transition_effect_kind not in _TRANSITION_EFFECT_KINDS
        or transition_effect_coverage not in _EFFECT_SCREEN_COVERAGES
        or primary_content_readability not in _PRIMARY_CONTENT_READABILITIES
        or not transition_relationship_is_valid
    ):
        raise _schema_error("publication_boundary_verification_schema_invalid")
    return (
        transient_transition_effect,
        cinematic_letterbox,
        event_staging,
        on_screen_dialogue_text_visible,
        visible_character_action,
    )


def _parse_candidate_frame_observations(
    raw_observations: list[object],
    *,
    frames: Mapping[str, FrameCandidate],
    catalog: SceneCatalog,
    context_relevance: ContextCueRelevance,
    cue_ids: tuple[str, ...],
) -> tuple[CandidateFrameObservation, ...]:
    """strictなframe別応答をdomain observationへ変換する。"""
    observations: list[CandidateFrameObservation] = []
    for raw_observation in raw_observations:
        if (
            not isinstance(raw_observation, dict)
            or set(raw_observation) != _FRAME_OBSERVATION_KEYS
        ):
            raise _schema_error("candidate_annotation_schema_invalid")
        frame_id = raw_observation.get("frame_id")
        scene_slug = raw_observation.get("scene_slug")
        content_kind = raw_observation.get("content_kind")
        interface_kind = raw_observation.get("interface_kind")
        prominent_event_portrait = raw_observation.get("prominent_event_portrait")
        cinematic_event_presentation = raw_observation.get(
            "cinematic_event_presentation"
        )
        visible_dialogue_text = raw_observation.get("on_screen_dialogue_text_visible")
        dialogue_text_presentation = raw_observation.get("dialogue_text_presentation")
        visible_action = raw_observation.get("visible_action")
        visible_character_or_enemy = raw_observation.get("visible_character_or_enemy")
        combat_action = raw_observation.get("combat_action")
        player_body_visibility = raw_observation.get("player_body_visibility")
        opponent_body_visibility = raw_observation.get("opponent_body_visibility")
        effect_only_frame = raw_observation.get("effect_only_frame")
        explanation_value = raw_observation.get("explanation_value")
        screen_text_kind = raw_observation.get("screen_text_kind")
        subject_visibility = raw_observation.get("primary_subject_visibility")
        transient_obstruction = raw_observation.get("transient_obstruction")
        spoiler_risk = raw_observation.get("spoiler_risk")
        spoiler_evidence = raw_observation.get("spoiler_evidence")
        if (
            not isinstance(frame_id, str)
            or not isinstance(scene_slug, str)
            or content_kind not in CANDIDATE_FRAME_CONTENT_KINDS
            or interface_kind not in CANDIDATE_INTERFACE_KINDS
            or not isinstance(prominent_event_portrait, bool)
            or not isinstance(cinematic_event_presentation, bool)
            or not isinstance(visible_dialogue_text, bool)
            or dialogue_text_presentation not in DIALOGUE_TEXT_PRESENTATIONS
            or not isinstance(visible_action, bool)
            or not isinstance(visible_character_or_enemy, bool)
            or not isinstance(combat_action, bool)
            or player_body_visibility not in CHARACTER_BODY_VISIBILITIES
            or opponent_body_visibility not in CHARACTER_BODY_VISIBILITIES
            or not isinstance(effect_only_frame, bool)
            or explanation_value not in EXPLANATION_VALUES
            or screen_text_kind not in SCREEN_TEXT_KINDS
            or subject_visibility not in PRIMARY_SUBJECT_VISIBILITIES
            or transient_obstruction not in TRANSIENT_OBSTRUCTIONS
            or spoiler_risk not in SPOILER_RISKS
            or not isinstance(spoiler_evidence, str)
        ):
            raise _schema_error("candidate_annotation_schema_invalid")
        if frame_id not in frames:
            raise _domain_error("candidate_annotation_representative_frame_unknown")
        if scene_slug not in catalog.slugs:
            raise _domain_error("candidate_annotation_scene_slug_unknown")
        typed_spoiler_risk = cast(SpoilerRisk, spoiler_risk)
        if not candidate_annotation_relationships_are_valid(
            context_relevance,
            cue_ids,
            typed_spoiler_risk,
            spoiler_evidence,
        ):
            raise _domain_error("candidate_annotation_relationship_invalid")
        try:
            observations.append(
                CandidateFrameObservation(
                    candidate=frames[frame_id],
                    scene_slug=scene_slug,
                    content_kind=cast(CandidateFrameContentKind, content_kind),
                    interface_kind=cast(CandidateInterfaceKind, interface_kind),
                    prominent_event_portrait=prominent_event_portrait,
                    cinematic_event_presentation=cinematic_event_presentation,
                    visible_dialogue_text=visible_dialogue_text,
                    dialogue_text_presentation=cast(
                        DialogueTextPresentation,
                        dialogue_text_presentation,
                    ),
                    visible_action=visible_action,
                    visible_character_or_enemy=visible_character_or_enemy,
                    combat_action=combat_action,
                    player_body_visibility=cast(
                        CharacterBodyVisibility,
                        player_body_visibility,
                    ),
                    opponent_body_visibility=cast(
                        CharacterBodyVisibility,
                        opponent_body_visibility,
                    ),
                    effect_only_frame=effect_only_frame,
                    explanation_value=cast(ExplanationValue, explanation_value),
                    screen_text_kind=cast(ScreenTextKind, screen_text_kind),
                    primary_subject_visibility=cast(
                        PrimarySubjectVisibility,
                        subject_visibility,
                    ),
                    transient_obstruction=cast(
                        TransientObstruction,
                        transient_obstruction,
                    ),
                    spoiler_risk=typed_spoiler_risk,
                    spoiler_evidence=spoiler_evidence,
                )
            )
        except ValueError:
            raise _domain_error("candidate_annotation_domain_invalid") from None
    return tuple(observations)


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
        "generation_options": _semantic_generation_options(),
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
        "generation_options": _semantic_generation_options(),
        "prompt_version": CANDIDATE_ANNOTATION_PROMPT_VERSION,
        "schema_version": CANDIDATE_ANNOTATION_SCHEMA_VERSION,
        "stage_contract_version": CANDIDATE_ANNOTATION_STAGE_CONTRACT_VERSION,
        "combat_encounter_verification_prompt_version": (
            COMBAT_ENCOUNTER_VERIFICATION_PROMPT_VERSION
        ),
        "combat_encounter_verification_schema_version": (
            COMBAT_ENCOUNTER_VERIFICATION_SCHEMA_VERSION
        ),
        "combat_encounter_verification_stage_contract_version": (
            COMBAT_ENCOUNTER_VERIFICATION_STAGE_CONTRACT_VERSION
        ),
        "combat_encounter_confirmation_prompt_version": (
            COMBAT_ENCOUNTER_CONFIRMATION_PROMPT_VERSION
        ),
        "combat_encounter_confirmation_schema_version": (
            COMBAT_ENCOUNTER_VERIFICATION_SCHEMA_VERSION
        ),
        "combat_encounter_confirmation_stage_contract_version": (
            COMBAT_ENCOUNTER_CONFIRMATION_STAGE_CONTRACT_VERSION
        ),
        "combat_visibility_verification_prompt_version": (
            COMBAT_VISIBILITY_VERIFICATION_PROMPT_VERSION
        ),
        "combat_visibility_verification_schema_version": (
            COMBAT_VISIBILITY_VERIFICATION_SCHEMA_VERSION
        ),
        "combat_visibility_verification_stage_contract_version": (
            COMBAT_VISIBILITY_VERIFICATION_STAGE_CONTRACT_VERSION
        ),
        "combat_visibility_confirmation_prompt_version": (
            COMBAT_VISIBILITY_CONFIRMATION_PROMPT_VERSION
        ),
        "combat_visibility_confirmation_schema_version": (
            COMBAT_VISIBILITY_VERIFICATION_SCHEMA_VERSION
        ),
        "combat_visibility_confirmation_stage_contract_version": (
            COMBAT_VISIBILITY_CONFIRMATION_STAGE_CONTRACT_VERSION
        ),
        "combat_visibility_edge_audit_prompt_version": (
            COMBAT_VISIBILITY_EDGE_AUDIT_PROMPT_VERSION
        ),
        "combat_visibility_edge_audit_schema_version": (
            COMBAT_VISIBILITY_VERIFICATION_SCHEMA_VERSION
        ),
        "combat_visibility_edge_audit_stage_contract_version": (
            COMBAT_VISIBILITY_EDGE_AUDIT_STAGE_CONTRACT_VERSION
        ),
        "publication_boundary_verification_prompt_version": (
            PUBLICATION_BOUNDARY_VERIFICATION_PROMPT_VERSION
        ),
        "publication_boundary_verification_schema_version": (
            PUBLICATION_BOUNDARY_VERIFICATION_SCHEMA_VERSION
        ),
        "publication_boundary_verification_stage_contract_version": (
            PUBLICATION_BOUNDARY_VERIFICATION_STAGE_CONTRACT_VERSION
        ),
        "retry_policy_version": RETRY_POLICY_VERSION,
    }


def _combat_encounter_verification_semantic_input(
    candidate: FrameCandidate,
    model: ResolvedModel,
    num_ctx: int,
    *,
    independently_confirm: bool = False,
) -> dict[str, object]:
    prompt_version = (
        COMBAT_ENCOUNTER_CONFIRMATION_PROMPT_VERSION
        if independently_confirm
        else COMBAT_ENCOUNTER_VERIFICATION_PROMPT_VERSION
    )
    stage_contract_version = (
        COMBAT_ENCOUNTER_CONFIRMATION_STAGE_CONTRACT_VERSION
        if independently_confirm
        else COMBAT_ENCOUNTER_VERIFICATION_STAGE_CONTRACT_VERSION
    )
    return {
        "frame_candidate": {
            "id": candidate.identifier,
            "image_sha256": hashlib.sha256(candidate.image_bytes).hexdigest(),
        },
        "model": {**model.semantic_input(), "num_ctx": num_ctx},
        "generation_options": _semantic_generation_options(),
        "prompt_version": prompt_version,
        "schema_version": COMBAT_ENCOUNTER_VERIFICATION_SCHEMA_VERSION,
        "stage_contract_version": stage_contract_version,
        "retry_policy_version": RETRY_POLICY_VERSION,
    }


def _combat_visibility_verification_semantic_input(
    candidate: FrameCandidate,
    model: ResolvedModel,
    num_ctx: int,
    *,
    independently_confirm: bool = False,
) -> dict[str, object]:
    prompt_version = (
        COMBAT_VISIBILITY_CONFIRMATION_PROMPT_VERSION
        if independently_confirm
        else COMBAT_VISIBILITY_VERIFICATION_PROMPT_VERSION
    )
    stage_contract_version = (
        COMBAT_VISIBILITY_CONFIRMATION_STAGE_CONTRACT_VERSION
        if independently_confirm
        else COMBAT_VISIBILITY_VERIFICATION_STAGE_CONTRACT_VERSION
    )
    return {
        "frame_candidate": {
            "id": candidate.identifier,
            "image_sha256": hashlib.sha256(candidate.image_bytes).hexdigest(),
        },
        "model": {**model.semantic_input(), "num_ctx": num_ctx},
        "generation_options": _semantic_generation_options(),
        "prompt_version": prompt_version,
        "schema_version": COMBAT_VISIBILITY_VERIFICATION_SCHEMA_VERSION,
        "stage_contract_version": stage_contract_version,
        "retry_policy_version": RETRY_POLICY_VERSION,
    }


def _combat_visibility_edge_audit_semantic_input(
    candidate: FrameCandidate,
    model: ResolvedModel,
    num_ctx: int,
) -> dict[str, object]:
    """四辺監査のcache identity入力を返す。"""
    return {
        "frame_candidate": {
            "id": candidate.identifier,
            "image_sha256": hashlib.sha256(candidate.image_bytes).hexdigest(),
        },
        "model": {**model.semantic_input(), "num_ctx": num_ctx},
        "generation_options": _semantic_generation_options(),
        "prompt_version": COMBAT_VISIBILITY_EDGE_AUDIT_PROMPT_VERSION,
        "schema_version": COMBAT_VISIBILITY_VERIFICATION_SCHEMA_VERSION,
        "stage_contract_version": COMBAT_VISIBILITY_EDGE_AUDIT_STAGE_CONTRACT_VERSION,
        "retry_policy_version": RETRY_POLICY_VERSION,
    }


def _publication_boundary_verification_semantic_input(
    candidate: FrameCandidate,
    model: ResolvedModel,
    num_ctx: int,
) -> dict[str, object]:
    return {
        "frame_candidate": {
            "id": candidate.identifier,
            "image_sha256": hashlib.sha256(candidate.image_bytes).hexdigest(),
        },
        "model": {**model.semantic_input(), "num_ctx": num_ctx},
        "generation_options": _semantic_generation_options(),
        "prompt_version": PUBLICATION_BOUNDARY_VERIFICATION_PROMPT_VERSION,
        "schema_version": PUBLICATION_BOUNDARY_VERIFICATION_SCHEMA_VERSION,
        "stage_contract_version": (
            PUBLICATION_BOUNDARY_VERIFICATION_STAGE_CONTRACT_VERSION
        ),
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
        "scene_kind": scene.scene_kind,
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
    if stage_kind == "combat_encounter_verification":
        return (
            COMBAT_ENCOUNTER_VERIFICATION_PROMPT_VERSION,
            COMBAT_ENCOUNTER_VERIFICATION_SCHEMA_VERSION,
            COMBAT_ENCOUNTER_VERIFICATION_STAGE_CONTRACT_VERSION,
        )
    if stage_kind == "combat_encounter_confirmation":
        return (
            COMBAT_ENCOUNTER_CONFIRMATION_PROMPT_VERSION,
            COMBAT_ENCOUNTER_VERIFICATION_SCHEMA_VERSION,
            COMBAT_ENCOUNTER_CONFIRMATION_STAGE_CONTRACT_VERSION,
        )
    if stage_kind == "combat_visibility_verification":
        return (
            COMBAT_VISIBILITY_VERIFICATION_PROMPT_VERSION,
            COMBAT_VISIBILITY_VERIFICATION_SCHEMA_VERSION,
            COMBAT_VISIBILITY_VERIFICATION_STAGE_CONTRACT_VERSION,
        )
    if stage_kind == "combat_visibility_confirmation":
        return (
            COMBAT_VISIBILITY_CONFIRMATION_PROMPT_VERSION,
            COMBAT_VISIBILITY_VERIFICATION_SCHEMA_VERSION,
            COMBAT_VISIBILITY_CONFIRMATION_STAGE_CONTRACT_VERSION,
        )
    if stage_kind == "combat_visibility_edge_audit":
        return (
            COMBAT_VISIBILITY_EDGE_AUDIT_PROMPT_VERSION,
            COMBAT_VISIBILITY_VERIFICATION_SCHEMA_VERSION,
            COMBAT_VISIBILITY_EDGE_AUDIT_STAGE_CONTRACT_VERSION,
        )
    if stage_kind == "publication_boundary_verification":
        return (
            PUBLICATION_BOUNDARY_VERIFICATION_PROMPT_VERSION,
            PUBLICATION_BOUNDARY_VERIFICATION_SCHEMA_VERSION,
            PUBLICATION_BOUNDARY_VERIFICATION_STAGE_CONTRACT_VERSION,
        )
    return (
        CANDIDATE_ANNOTATION_PROMPT_VERSION,
        CANDIDATE_ANNOTATION_SCHEMA_VERSION,
        CANDIDATE_ANNOTATION_STAGE_CONTRACT_VERSION,
    )


def _merge_candidate_diagnostics(
    primary: VisionInferenceDiagnostics,
    verification: VisionInferenceDiagnostics,
) -> VisionInferenceDiagnostics:
    """注釈と条件付き専用確認のsafe計測値を一つのStage診断へ集約する。"""
    if (
        primary.model_name != verification.model_name
        or primary.model_identity != verification.model_identity
        or primary.runtime_identity != verification.runtime_identity
        or primary.retry_policy_version != verification.retry_policy_version
    ):
        msg = "Candidate Annotationの推論診断identityが一致しません"
        raise ValueError(msg)
    return replace(
        primary,
        attempt_count=primary.attempt_count + verification.attempt_count,
        validation_code=primary.validation_code or verification.validation_code,
        duration_seconds=primary.duration_seconds + verification.duration_seconds,
        prompt_eval_count=_sum_optional_counts(
            primary.prompt_eval_count,
            verification.prompt_eval_count,
        ),
        eval_count=_sum_optional_counts(
            primary.eval_count,
            verification.eval_count,
        ),
        done_reason=verification.done_reason or primary.done_reason,
    )


def _is_publishable_combat_visibility(
    observation: tuple[CharacterBodyVisibility, OpponentBodyFraming, bool],
) -> bool:
    """敵本体が明瞭で構図内に収まりeffectだけではない場合だけ許可する。"""
    opponent_body_visibility, opponent_body_framing, effect_only_frame = observation
    return (
        opponent_body_visibility == "clear"
        and opponent_body_framing == "complete"
        and not effect_only_frame
    )


def _is_consistent_noncombat_or_publishable_combat_visibility(
    first: tuple[CharacterBodyVisibility, OpponentBodyFraming, bool],
    confirmation: tuple[CharacterBodyVisibility, OpponentBodyFraming, bool],
) -> bool:
    """二回とも敵不在、または二回とも掲載可能な戦闘の場合だけ許可する。"""
    observations = (first, confirmation)
    opponent_is_observed = any(
        opponent_body_visibility != "absent" or opponent_body_framing != "absent"
        for opponent_body_visibility, opponent_body_framing, _ in observations
    )
    if opponent_is_observed:
        return all(_is_publishable_combat_visibility(item) for item in observations)
    return all(not effect_only_frame for _, _, effect_only_frame in observations)


def _sum_optional_counts(left: int | None, right: int | None) -> int | None:
    if left is None or right is None:
        return None
    return left + right


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
