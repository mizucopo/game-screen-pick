"""Ollama structured outputsを使うVisionRuntime adapter。"""

import base64
import copy
import hashlib
import io
import json
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from fractions import Fraction
from functools import partial
from typing import Literal, TypeAlias, TypeVar, cast
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image

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
from ..models.combat_encounter_basis import (
    COMBAT_ENCOUNTER_BASES,
    CombatEncounterBasis,
    combat_encounter_classification_is_valid,
)
from ..models.combat_encounter_kind import (
    COMBAT_ENCOUNTER_KINDS,
    CombatEncounterKind,
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
from .detect_cinematic_letterbox import (
    CINEMATIC_LETTERBOX_DETECTION_VERSION,
    has_cinematic_letterbox,
)
from .vision_contract import (
    CANDIDATE_ANNOTATION_PROMPT_VERSION,
    CANDIDATE_ANNOTATION_RELATIONSHIP_REPAIR_EVIDENCE_MAX_LENGTH,
    CANDIDATE_ANNOTATION_RELATIONSHIP_REPAIR_NUM_PREDICT,
    CANDIDATE_ANNOTATION_RELATIONSHIP_REPAIR_PROMPT_VERSION,
    CANDIDATE_ANNOTATION_RELATIONSHIP_REPAIR_SCHEMA_VERSION,
    CANDIDATE_ANNOTATION_RELATIONSHIP_REPAIR_STAGE_CONTRACT_VERSION,
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
    COMBAT_VISIBILITY_EDGE_AUDIT_SCHEMA,
    COMBAT_VISIBILITY_EDGE_AUDIT_SCHEMA_VERSION,
    COMBAT_VISIBILITY_EDGE_AUDIT_STAGE_CONTRACT_VERSION,
    COMBAT_VISIBILITY_EDGE_STRIP_VERSION,
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
CandidateDraftValidator = Callable[[Mapping[str, object]], None]
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
ResponseStageKind = StageKind | Literal["candidate_annotation_relationship_repair"]
OpponentBodyFraming = Literal["complete", "edge_cropped", "occluded", "absent"]
CombatVisibilityObservation: TypeAlias = tuple[
    CharacterBodyVisibility,
    CharacterBodyVisibility,
    OpponentBodyFraming,
    bool,
]
CombatEncounterClassification: TypeAlias = tuple[
    CombatEncounterKind,
    CombatEncounterBasis,
]


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
    "scene_catalog_match",
    "content_kind",
    "interface_kind",
    "prominent_event_portrait",
    "cinematic_event_presentation",
    "on_screen_dialogue_text_visible",
    "dialogue_text_presentation",
    "visible_action",
    "visible_character_or_enemy",
    "combat_encounter_kind",
    "combat_encounter_basis",
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
_COMBAT_VISIBILITY_EDGE_NAMES = ("top", "bottom", "left", "right")
_COMBAT_VISIBILITY_EDGE_OBSERVATION_KEYS = {
    "edge",
    "opponent_body_present",
    "opponent_body_reaches_outer_edge",
}
_COMBAT_ENCOUNTER_VERIFICATION_KEYS = {
    "combat_encounter_kind",
    "combat_encounter_basis",
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
    "sceneはブログで役割が異なる視覚・内容のまとまりとして作ります。"
    "slug、display_name、descriptionは、そのsceneへ後から分類されるどの画像にも"
    "当てはまる再利用可能なカテゴリ表現にします。一部の代表画像だけから推測した"
    "町・ダンジョンなどの場所、固有人物、物語上の結果をscene全体へ断定せず、"
    "会話イベント・boss戦・地図画面のような視覚・操作上の役割を優先します。\n"
)
_CANDIDATE_FRAME_DIRECT_OBSERVATION_INSTRUCTION = (
    "この画像だけに実際に見えるものを最初に観測してください。"
    "scene_catalog_matchは、選んだscene_slugのdisplay_nameとdescriptionに含まれる"
    "具体的な場所・人物・出来事まで、この画像だけで確認できる場合だけtrueです。"
    "Scene Kindや会話・戦闘という大分類だけが一致する場合、または音声・Context Cue"
    "から補わなければ一致しない場合はfalseです。"
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
    "説明文は使いません。combat_encounter_kindはnot_combat、ordinary、major、"
    "uncertainから選びます。敵またはboss固有の名前とHP・status barがある、または"
    "player本体と攻撃相手本体が戦闘中だと画面から分かる場合はnot_combat以外です。"
    "敵本体が画面端で切れる、エフェクトに隠れる、画面外にいる場合でも、敵固有の"
    "名前とHP・status barがあれば戦闘です。player自身の通常HP、portrait、操作button、"
    "minimapだけならnot_combatです。combat_encounter_basisはnone、"
    "ordinary_opponent_presentation、ordinary_encounter_presentation、"
    "major_opponent_presentation、major_encounter_presentation、ambiguousから選び、"
    "combat_encounter_kindと必ず整合させます。一般敵の群れ・編成が画像から分かる、"
    "または通常・反復可能な遭遇だと直接分かる積極的な画像内根拠がある場合だけ"
    "ordinaryです。主要な相手の外見、boss専用表示、特別な演出や構図が直接見える"
    "場合はmajorです。戦闘は見えてもどちらの積極的根拠もない場合はuncertainかつ"
    "ambiguousです。根拠が競合する場合もuncertainです。敵名やHP・status barだけでは"
    "ordinaryにもmajorにもせず、主要戦闘の根拠がないことだけをordinaryの根拠に"
    "しません。combat_encounter_kindは物語上の"
    "spoiler_riskとは独立して決めます。"
    "combat_encounter_evidenceはnone、enemy_status_ui、opposing_bodies、bothから"
    "選びます。not_combatならnone、それ以外ならnone以外です。"
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
    "添付した4画像は、同じ戦闘スクリーンショットから外周30%を切り出した診断画像で、"
    "順番はtop、bottom、left、rightです。各診断画像では、そのedge名と同じ側だけが"
    "元スクリーンショットの実際の外端で、反対側は診断用の内側crop境界です。"
    "操作player、HUD、敵名、HP bar、光、hit effect、影、背景ではなく、playerが攻撃する"
    "相手の生物・monster・boss本体だけを観測してください。opponent_body_presentは、"
    "その本体の主要な輪郭を判別できる場合だけtrueです。"
    "opponent_body_reaches_outer_edgeは、その本体の主要な輪郭が元スクリーンショットの"
    "実際の外端に触れる、または外へ続く場合だけtrueです。診断用の内側crop境界に触れる"
    "ことは数えません。edgesをtop、bottom、left、rightの順で必ず4件返してください。"
    "推論過程や説明文は返しません。"
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
    "scene_slugは最も近いScene Catalog entryから選び、scene_catalog_matchはその"
    "entryの具体的な表示名と説明が画像だけで裏付けられるかを返します。場所名・"
    "人物名・物語上の出来事の一部でも画像から確認できなければfalseです。音声、"
    "Context Cue、Video Set Progressを一致の根拠にしません。"
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
    "combat_encounter_kindはnot_combat・ordinary・major・uncertainから選び、"
    "combat_encounter_basisと必ず整合させます。not_combatは戦闘が見えない場面で"
    "basis=noneです。一般敵の群れ・編成が画像から分かるordinary_opponent_presentation、"
    "または通常・反復可能な遭遇だと直接分かるordinary_encounter_presentationという"
    "積極的な画像内根拠がある場合だけordinaryです。majorは主要な相手の外見を直接示す"
    "major_opponent_presentation、またはboss専用表示・特別な演出や構図を示す"
    "major_encounter_presentationを持つ戦闘です。戦闘は見えてもどちらの積極的根拠も"
    "ない場合はuncertainかつbasis=ambiguousです。根拠が競合する場合もuncertainです。"
    "敵名やHP・status barだけではordinaryにもmajorにもせず、主要戦闘を示す直接根拠が"
    "ないことだけをordinaryの根拠にしません。"
    "combat_encounter_kindは物語上のspoiler_riskとは独立して決めます。"
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
            *,
            count_as_candidate_response: bool,
        ) -> tuple[CandidateAnnotation, bool, bool, bool, bool, bool]:
            nonlocal candidate_response_count
            (
                annotation,
                redacted,
                requires_dialogue_verification,
                requires_combat_verification,
                requires_combat_encounter_verification,
                requires_publication_verification,
                scene_is_combat,
            ) = _parse_candidate_annotation(value, request, catalog)
            if count_as_candidate_response:
                candidate_response_count += 1
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
                scene_is_combat,
            )

        def parse_candidate_response(
            value: Mapping[str, object],
        ) -> tuple[CandidateAnnotation, bool, bool, bool, bool, bool]:
            return parse_candidate(value, count_as_candidate_response=True)

        def parse_candidate_relationship_repair(
            value: Mapping[str, object],
        ) -> tuple[CandidateAnnotation, bool, bool, bool, bool, bool]:
            return parse_candidate(value, count_as_candidate_response=True)

        (
            (
                annotation,
                free_text_redacted,
                requires_combat_verification,
                requires_combat_encounter_verification,
                requires_publication_verification,
                scene_is_combat,
            ),
            diagnostics,
        ) = self._infer(
            stage_kind="candidate_annotation",
            request_fingerprint=_fingerprint(semantic_input),
            payload=_candidate_payload(request, catalog, model, num_ctx),
            parser=parse_candidate_response,
            model=model,
            image_count=len(request.frame_candidates),
            context_cue_count=len(request.context_cues),
            candidate_request=request,
            candidate_draft_validator=partial(
                _validate_candidate_annotation_repair_draft,
                request=request,
                catalog=catalog,
            ),
            candidate_relationship_repair_parser=parse_candidate_relationship_repair,
        )
        cinematic_letterbox_detected = has_cinematic_letterbox(
            annotation.candidate.image_bytes
        )
        requires_publication_verification = requires_publication_verification or (
            annotation.explanation_value != "none" and cinematic_letterbox_detected
        )
        combat_scene = scene_is_combat
        verified_combat_encounter_kind = annotation.combat_encounter_kind
        verified_combat_encounter_basis = annotation.combat_encounter_basis
        if requires_combat_encounter_verification:
            verification_input = _combat_encounter_verification_semantic_input(
                annotation.candidate,
                model,
                num_ctx,
            )
            (
                (
                    verified_combat_encounter_kind,
                    verified_combat_encounter_basis,
                ),
                verification_diagnostics,
            ) = self._infer(
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
            if verified_combat_encounter_kind == "not_combat":
                confirmation_input = _combat_encounter_verification_semantic_input(
                    annotation.candidate,
                    model,
                    num_ctx,
                    independently_confirm=True,
                )
                (
                    (
                        verified_combat_encounter_kind,
                        verified_combat_encounter_basis,
                    ),
                    confirmation_diagnostics,
                ) = self._infer(
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
            if verified_combat_encounter_kind == "not_combat" and combat_scene:
                annotation = replace(annotation, explanation_value="none")
            requires_combat_verification = (
                verified_combat_encounter_kind != "not_combat"
            )
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
                opponent_reaches_outer_edge, edge_audit_diagnostics = self._infer(
                    stage_kind="combat_visibility_edge_audit",
                    request_fingerprint=_fingerprint(edge_audit_input),
                    payload=_combat_visibility_edge_audit_payload(
                        annotation.candidate,
                        model,
                        num_ctx,
                    ),
                    parser=_parse_combat_visibility_edge_audit,
                    model=model,
                    image_count=4,
                    context_cue_count=0,
                )
                diagnostics = _merge_candidate_diagnostics(
                    diagnostics,
                    edge_audit_diagnostics,
                )
                visibility_is_acceptable = not opponent_reaches_outer_edge
            if not visibility_is_acceptable:
                annotation = replace(annotation, explanation_value="none")
            elif combat_is_consistently_publishable:
                if verified_combat_encounter_kind == "not_combat":
                    verified_combat_encounter_kind = "uncertain"
                    verified_combat_encounter_basis = "ambiguous"
                annotation = replace(
                    annotation,
                    combat_encounter_kind=verified_combat_encounter_kind,
                    combat_encounter_basis=verified_combat_encounter_basis,
                )
        if annotation.explanation_value != "none" and requires_publication_verification:
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
                (cinematic_letterbox or cinematic_letterbox_detected)
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
        candidate_request: CandidateAnnotationRequest | None = None,
        candidate_draft_validator: CandidateDraftValidator | None = None,
        candidate_relationship_repair_parser: (
            InferenceParser[InferenceValue] | None
        ) = None,
    ) -> tuple[InferenceValue, VisionInferenceDiagnostics]:
        """同じsemantic入力を検証し、関係修復併用時だけ最大3 requestを許可する。"""
        started_at = time.monotonic()
        previous_validation_code: str | None = None
        repair_code: str | None = None
        candidate_relationship_draft: Mapping[str, object] | None = None
        for attempt in (1, 2, 3):
            relationship_repair = candidate_relationship_draft is not None
            decoded: Mapping[str, object] | None = None
            try:
                attempt_payload = (
                    _candidate_relationship_repair_payload(
                        payload,
                        candidate_relationship_draft,
                        candidate_request,
                    )
                    if candidate_relationship_draft is not None
                    else _with_repair_code(payload, repair_code)
                )
                self._require_frozen_model_state(model)
                response = self._request(attempt_payload)
                self._require_frozen_model_state(model)
                decoded = _decode_content(
                    response,
                    (
                        "candidate_annotation_relationship_repair"
                        if relationship_repair
                        else stage_kind
                    ),
                )
                active_parser = (
                    candidate_relationship_repair_parser
                    if relationship_repair
                    and candidate_relationship_repair_parser is not None
                    else parser
                )
                value = active_parser(
                    _merge_candidate_relationship_repair(
                        candidate_relationship_draft,
                        decoded,
                        candidate_request,
                    )
                    if candidate_relationship_draft is not None
                    else decoded
                )
            except VisionRuntimeError as error:
                if (
                    stage_kind == "candidate_annotation"
                    and error.validation_code
                    == "candidate_annotation_dialogue_visibility_unverified"
                    and attempt < 3
                ):
                    previous_validation_code = error.validation_code
                    candidate_relationship_draft = None
                    repair_code = _repair_validation_code(error)
                    self._sleeper(error.retry_after_seconds)
                    continue
                if relationship_repair:
                    error = _relationship_repair_error(error)
                elif (
                    error.validation_code == "candidate_annotation_relationship_invalid"
                    and decoded is not None
                    and candidate_draft_validator is not None
                ):
                    try:
                        candidate_draft_validator(decoded)
                    except VisionRuntimeError as draft_error:
                        error = draft_error
                if attempt >= 2 or error.reason not in _RETRYABLE_REASONS:
                    raise VisionRuntimeError(
                        error.reason,
                        validation_code=error.validation_code,
                        attempt_count=attempt,
                    ) from None
                previous_validation_code = error.validation_code
                if (
                    stage_kind == "candidate_annotation"
                    and error.validation_code
                    == "candidate_annotation_relationship_invalid"
                    and decoded is not None
                ):
                    candidate_relationship_draft = decoded
                    repair_code = None
                else:
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


def _candidate_relationship_repair_payload(
    payload: Mapping[str, object],
    draft: Mapping[str, object],
    request: CandidateAnnotationRequest | None,
) -> dict[str, object]:
    """関係違反の従属fieldだけを再生成するrequestを返す。"""
    if request is None:
        raise _domain_error("candidate_annotation_relationship_repair_domain_invalid")
    evidence_repairs = _spoiler_evidence_repairs(draft)
    repair_context = _supporting_context_cue_ids_need_repair(draft)
    if not evidence_repairs and not repair_context:
        raise _domain_error("candidate_annotation_relationship_repair_domain_invalid")
    repair_ids = tuple(frame_id for frame_id, _risk in evidence_repairs)
    raw_messages = payload.get("messages")
    raw_options = payload.get("options")
    if not isinstance(raw_messages, list) or not isinstance(raw_options, dict):
        raise _domain_error("candidate_annotation_relationship_repair_domain_invalid")
    message_ids = (
        {item.identifier for item in request.frame_candidates}
        if repair_context
        else set(repair_ids)
    )
    frame_messages = [
        copy.deepcopy(message)
        for message in raw_messages[:-1]
        if isinstance(message, dict)
        and isinstance(message.get("content"), str)
        and any(
            message["content"].startswith(f"frame_candidate_id={frame_id}。")
            for frame_id in message_ids
        )
    ]
    if len(frame_messages) != len(message_ids):
        raise _domain_error("candidate_annotation_relationship_repair_domain_invalid")
    required: list[str] = []
    properties: dict[str, object] = {}
    semantic_request: dict[str, object] = {}
    instructions: list[str] = []
    if repair_context:
        context_relevance = cast(str, draft["context_relevance"])
        cue_ids = [item.identifier for item in request.context_cues]
        cue_schema: dict[str, object] = {
            "type": "array",
            "uniqueItems": True,
            "maxItems": len(cue_ids),
            "items": {"type": "string", "enum": cue_ids},
        }
        if context_relevance in {"weak", "strong"}:
            cue_schema["minItems"] = 1
        else:
            cue_schema["maxItems"] = 0
        required.append("supporting_context_cue_ids")
        properties["supporting_context_cue_ids"] = cue_schema
        semantic_request["frozen_context_relevance"] = context_relevance
        semantic_request["context_cues"] = [
            {
                "id": cue.identifier,
                "start": _fraction_value(cue.start),
                "end": _fraction_value(cue.end),
                "text": cue.text,
            }
            for cue in request.context_cues
        ]
        instructions.append(
            "context_relevanceを変更しません。frozen_context_relevanceがnoneまたは"
            "unavailableならsupporting_context_cue_idsは空配列、weakまたはstrong"
            "なら入力Context Cue IDを1件以上、重複なしで返します。"
        )
    if evidence_repairs:
        evidence_schema: dict[str, object] = {
            "type": "string",
            "maxLength": (CANDIDATE_ANNOTATION_RELATIONSHIP_REPAIR_EVIDENCE_MAX_LENGTH),
        }
        risks = {spoiler_risk for _frame_id, spoiler_risk in evidence_repairs}
        if "none" not in risks:
            evidence_schema["minLength"] = 1
        elif risks == {"none"}:
            evidence_schema["maxLength"] = 0
        required.append("frame_observations")
        properties["frame_observations"] = {
            "type": "array",
            "minItems": len(evidence_repairs),
            "maxItems": len(evidence_repairs),
            "uniqueItems": True,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["frame_id", "spoiler_evidence"],
                "properties": {
                    "frame_id": {
                        "type": "string",
                        "enum": list(repair_ids),
                    },
                    "spoiler_evidence": evidence_schema,
                },
            },
        }
        semantic_request["frame_observations"] = [
            {"frame_id": frame_id, "frozen_spoiler_risk": spoiler_risk}
            for frame_id, spoiler_risk in evidence_repairs
        ]
        instructions.append(
            "spoiler_riskを変更しません。frozen_spoiler_riskがnoneなら"
            "spoiler_evidenceは空文字列、それ以外なら各frameの画素だけから"
            "判断できる空でない根拠を160文字以内の1文で返します。"
        )
        if repair_context:
            instructions.append(
                "Context Cue本文をspoiler_evidenceへ引用しません。正規化後3〜5文字の"
                "Cueは全文、6文字以上のCueは6文字以上の連続部分も再出力しません。"
            )
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": properties,
    }
    instruction = (
        "前回のCandidate Annotationはschemaに適合しましたが、"
        "validation_code=candidate_annotation_relationship_invalidでした。"
        "前回の応答本文は参照せず、同じ入力から関係違反の従属fieldだけを"
        "再生成します。"
        + "".join(instructions)
        + "repair対象以外のfield、推論過程、逐語的な画面文は出力しません。\n"
        + json.dumps(semantic_request, ensure_ascii=False, sort_keys=True)
    )
    options = cast(dict[str, object], copy.deepcopy(raw_options))
    options["num_predict"] = CANDIDATE_ANNOTATION_RELATIONSHIP_REPAIR_NUM_PREDICT
    return {
        "model": payload.get("model"),
        "stream": False,
        "think": False,
        "format": schema,
        "options": options,
        "messages": [
            *frame_messages,
            {"role": "user", "content": instruction},
        ],
    }


def _spoiler_evidence_repairs(
    draft: Mapping[str, object],
) -> tuple[tuple[str, SpoilerRisk], ...]:
    """Spoiler Riskとevidenceが不整合なframeを入力順で返す。"""
    raw_observations = draft.get("frame_observations")
    if not isinstance(raw_observations, list):
        raise _domain_error("candidate_annotation_relationship_repair_domain_invalid")
    repairs: list[tuple[str, SpoilerRisk]] = []
    for observation in raw_observations:
        if not isinstance(observation, dict):
            raise _domain_error(
                "candidate_annotation_relationship_repair_domain_invalid"
            )
        frame_id = observation.get("frame_id")
        spoiler_risk = observation.get("spoiler_risk")
        spoiler_evidence = observation.get("spoiler_evidence")
        if (
            not isinstance(frame_id, str)
            or spoiler_risk not in SPOILER_RISKS
            or not isinstance(spoiler_evidence, str)
        ):
            raise _domain_error(
                "candidate_annotation_relationship_repair_domain_invalid"
            )
        relationship_is_valid = (
            not spoiler_evidence
            if spoiler_risk == "none"
            else bool(spoiler_evidence.strip())
        )
        if not relationship_is_valid:
            repairs.append((frame_id, cast(SpoilerRisk, spoiler_risk)))
    return tuple(repairs)


def _supporting_context_cue_ids_need_repair(
    draft: Mapping[str, object],
) -> bool:
    """凍結済みrelevanceと参照Cue IDの関係違反を返す。"""
    context_relevance = draft.get("context_relevance")
    cue_ids = draft.get("supporting_context_cue_ids")
    if (
        context_relevance not in CONTEXT_CUE_RELEVANCES
        or not isinstance(cue_ids, list)
        or not all(isinstance(item, str) for item in cue_ids)
    ):
        raise _domain_error("candidate_annotation_relationship_repair_domain_invalid")
    return bool(cue_ids) != (context_relevance in {"weak", "strong"})


def _merge_candidate_relationship_repair(
    draft: Mapping[str, object],
    repair: Mapping[str, object],
    request: CandidateAnnotationRequest | None,
) -> Mapping[str, object]:
    """許可した従属fieldだけをdraftへ統合する。"""
    if request is None:
        raise _domain_error("candidate_annotation_relationship_repair_domain_invalid")
    expected_evidence = _spoiler_evidence_repairs(draft)
    repair_context = _supporting_context_cue_ids_need_repair(draft)
    expected_keys: set[str] = set()
    if repair_context:
        expected_keys.add("supporting_context_cue_ids")
    if expected_evidence:
        expected_keys.add("frame_observations")
    if set(repair) != expected_keys:
        raise _schema_error("candidate_annotation_relationship_repair_schema_invalid")
    merged = cast(dict[str, object], copy.deepcopy(draft))
    if repair_context:
        raw_cue_ids = repair.get("supporting_context_cue_ids")
        if not isinstance(raw_cue_ids, list) or not all(
            isinstance(item, str) for item in raw_cue_ids
        ):
            raise _schema_error(
                "candidate_annotation_relationship_repair_schema_invalid"
            )
        cue_ids = cast(list[str], raw_cue_ids)
        context_relevance = cast(str, draft["context_relevance"])
        available_ids = {item.identifier for item in request.context_cues}
        if (
            len(cue_ids) != len(set(cue_ids))
            or not set(cue_ids).issubset(available_ids)
            or bool(cue_ids) != (context_relevance in {"weak", "strong"})
        ):
            raise _domain_error(
                "candidate_annotation_relationship_repair_domain_invalid"
            )
        merged["supporting_context_cue_ids"] = cue_ids
    if expected_evidence:
        raw_repairs = repair.get("frame_observations")
        if not isinstance(raw_repairs, list):
            raise _schema_error(
                "candidate_annotation_relationship_repair_schema_invalid"
            )
        if len(raw_repairs) != len(expected_evidence):
            raise _domain_error(
                "candidate_annotation_relationship_repair_domain_invalid"
            )
        values: dict[str, str] = {}
        for raw_repair, (expected_frame_id, spoiler_risk) in zip(
            raw_repairs,
            expected_evidence,
            strict=True,
        ):
            if not isinstance(raw_repair, dict) or set(raw_repair) != {
                "frame_id",
                "spoiler_evidence",
            }:
                raise _schema_error(
                    "candidate_annotation_relationship_repair_schema_invalid"
                )
            frame_id = raw_repair.get("frame_id")
            evidence = raw_repair.get("spoiler_evidence")
            if not isinstance(frame_id, str) or not isinstance(evidence, str):
                raise _schema_error(
                    "candidate_annotation_relationship_repair_schema_invalid"
                )
            evidence_is_valid = (
                not evidence if spoiler_risk == "none" else bool(evidence.strip())
            )
            if (
                frame_id != expected_frame_id
                or not evidence_is_valid
                or len(evidence)
                > CANDIDATE_ANNOTATION_RELATIONSHIP_REPAIR_EVIDENCE_MAX_LENGTH
            ):
                raise _domain_error(
                    "candidate_annotation_relationship_repair_domain_invalid"
                )
            values[frame_id] = evidence
        merged_observations = cast(
            list[dict[str, object]],
            merged["frame_observations"],
        )
        for observation in merged_observations:
            frame_id = cast(str, observation["frame_id"])
            if frame_id in values:
                observation["spoiler_evidence"] = values[frame_id]
    return merged


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
    """敵本体が外端へ続くfalse positiveを4辺stripで監査するrequestを返す。"""
    return {
        "model": model.configured_name,
        "stream": False,
        "think": False,
        "format": COMBAT_VISIBILITY_EDGE_AUDIT_SCHEMA,
        "options": _generation_options(num_ctx),
        "messages": [
            {
                "role": "user",
                "content": _COMBAT_VISIBILITY_EDGE_AUDIT_INSTRUCTION,
                "images": [
                    base64.b64encode(image_bytes).decode()
                    for image_bytes in _combat_visibility_edge_strips(
                        candidate.image_bytes
                    )
                ],
            }
        ],
    }


def _combat_visibility_edge_strips(image_bytes: bytes) -> tuple[bytes, ...]:
    """画像のtop、bottom、left、right外周30%を決定的なJPEGで返す。"""
    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            image = source.convert("RGB")
        width, height = image.size
        boxes = (
            (0, 0, width, max(1, height * 3 // 10)),
            (0, min(height - 1, height * 7 // 10), width, height),
            (0, 0, max(1, width * 3 // 10), height),
            (min(width - 1, width * 7 // 10), 0, width, height),
        )
        strips: list[bytes] = []
        for box in boxes:
            output = io.BytesIO()
            image.crop(box).save(output, format="JPEG", quality=95)
            strips.append(output.getvalue())
    except (OSError, ValueError):
        raise VisionRuntimeError(
            VisionRuntimeFailureReason.INVALID_REQUEST,
            validation_code="combat_visibility_edge_audit_image_invalid",
        ) from None
    return tuple(strips)


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
    response: Mapping[str, object], stage_kind: ResponseStageKind
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
            display_name = "その他"
            description = "共有Scene Catalogの他sceneに分類できない場面"
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


def _validate_candidate_annotation_repair_draft(
    value: Mapping[str, object],
    *,
    request: CandidateAnnotationRequest,
    catalog: SceneCatalog,
) -> None:
    """relationship以外の全fieldが部分repair前にvalidであることを検証する。"""
    normalized = cast(dict[str, object], copy.deepcopy(value))
    context_relevance = normalized.get("context_relevance")
    cue_ids = normalized.get("supporting_context_cue_ids")
    available_cue_ids = tuple(item.identifier for item in request.context_cues)
    if (
        context_relevance in CONTEXT_CUE_RELEVANCES
        and isinstance(cue_ids, list)
        and all(isinstance(item, str) for item in cue_ids)
        and len(cue_ids) == len(set(cue_ids))
        and candidate_annotation_context_is_valid(
            context_relevance,
            tuple(cast(list[str], cue_ids)),
            available_cue_ids,
        )
    ):
        normalized["supporting_context_cue_ids"] = (
            [available_cue_ids[0]]
            if context_relevance in {"weak", "strong"} and available_cue_ids
            else []
        )
    raw_observations = normalized.get("frame_observations")
    if isinstance(raw_observations, list):
        for raw_observation in raw_observations:
            if not isinstance(raw_observation, dict):
                continue
            spoiler_risk = raw_observation.get("spoiler_risk")
            spoiler_evidence = raw_observation.get("spoiler_evidence")
            if spoiler_risk in SPOILER_RISKS and isinstance(spoiler_evidence, str):
                raw_observation["spoiler_evidence"] = (
                    "" if spoiler_risk == "none" else "relationship validation"
                )
    _parse_candidate_annotation(normalized, request, catalog)


def _parse_candidate_annotation(
    value: Mapping[str, object],
    request: CandidateAnnotationRequest,
    catalog: SceneCatalog,
) -> tuple[CandidateAnnotation, bool, bool, bool, bool, bool, bool]:
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
        annotation_scene_slug = (
            selected.scene_slug if selected.scene_catalog_match else "other"
        )
        annotation_scene = catalog.for_slug(annotation_scene_slug)
        annotation_summary = (
            f"{annotation_scene.display_name}の{content_label}"
            if selected.scene_catalog_match
            else content_label
        )
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
            scene_slug=annotation_scene_slug,
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
                scene_slug=annotation_scene_slug,
                blog_image_type=selected.blog_image_type,
                explanation_value=selected.effective_explanation_value,
                frame_choice_reason=frame_choice_reason,
                screen_text_kind=selected.effective_screen_text_kind,
                context_relevance=typed_context_relevance,
                supporting_context_cue_ids=typed_cue_ids,
                spoiler_risk=selected.spoiler_risk,
                spoiler_evidence=spoiler_evidence,
                combat_encounter_kind=selected.combat_encounter_kind,
                combat_encounter_basis=selected.combat_encounter_basis,
            ),
            free_text_redacted,
            requires_dialogue_verification,
            requires_combat_verification,
            requires_combat_encounter_verification,
            requires_publication_verification,
            scene.scene_kind == "combat",
        )
    except ValueError:
        raise _domain_error("candidate_annotation_domain_invalid") from None


def _parse_combat_encounter_verification(
    value: Mapping[str, object],
) -> CombatEncounterClassification:
    """戦闘種別と根拠enumの関係を検証して戦闘種別を返す。"""
    if set(value) != _COMBAT_ENCOUNTER_VERIFICATION_KEYS:
        raise _schema_error("combat_encounter_verification_schema_invalid")
    combat_encounter_kind = value.get("combat_encounter_kind")
    combat_encounter_basis = value.get("combat_encounter_basis")
    combat_encounter_evidence = value.get("combat_encounter_evidence")
    combat_is_visible = combat_encounter_kind != "not_combat"
    evidence_is_present = combat_encounter_evidence != "none"
    if (
        combat_encounter_kind not in COMBAT_ENCOUNTER_KINDS
        or combat_encounter_basis not in COMBAT_ENCOUNTER_BASES
        or combat_encounter_evidence not in _COMBAT_ENCOUNTER_EVIDENCE
        or combat_is_visible != evidence_is_present
    ):
        raise _schema_error("combat_encounter_verification_schema_invalid")
    typed_kind = combat_encounter_kind
    typed_basis = combat_encounter_basis
    if not combat_encounter_classification_is_valid(typed_kind, typed_basis):
        raise _schema_error("combat_encounter_verification_schema_invalid")
    return typed_kind, typed_basis


def _parse_combat_visibility_verification(
    value: Mapping[str, object],
) -> CombatVisibilityObservation:
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
    return (
        player_body_visibility,
        opponent_body_visibility,
        opponent_body_framing,
        effect_only_frame,
    )


def _parse_combat_visibility_edge_audit(value: Mapping[str, object]) -> bool:
    """4辺の直接観測を検証し、敵本体が外端へ到達するかだけを返す。"""
    raw_edges = value.get("edges")
    if set(value) != {"edges"} or not isinstance(raw_edges, list):
        raise _schema_error("combat_visibility_edge_audit_schema_invalid")
    if len(raw_edges) != len(_COMBAT_VISIBILITY_EDGE_NAMES):
        raise _schema_error("combat_visibility_edge_audit_schema_invalid")
    opponent_reaches_outer_edge = False
    for expected_edge, raw_edge in zip(
        _COMBAT_VISIBILITY_EDGE_NAMES,
        raw_edges,
        strict=True,
    ):
        if (
            not isinstance(raw_edge, dict)
            or set(raw_edge) != _COMBAT_VISIBILITY_EDGE_OBSERVATION_KEYS
            or raw_edge.get("edge") != expected_edge
        ):
            raise _schema_error("combat_visibility_edge_audit_schema_invalid")
        opponent_body_present = raw_edge.get("opponent_body_present")
        reaches_outer_edge = raw_edge.get("opponent_body_reaches_outer_edge")
        if (
            not isinstance(opponent_body_present, bool)
            or not isinstance(reaches_outer_edge, bool)
            or (reaches_outer_edge and not opponent_body_present)
        ):
            raise _schema_error("combat_visibility_edge_audit_schema_invalid")
        opponent_reaches_outer_edge |= opponent_body_present and reaches_outer_edge
    return opponent_reaches_outer_edge


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
        scene_catalog_match = raw_observation.get("scene_catalog_match")
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
        combat_encounter_kind = raw_observation.get("combat_encounter_kind")
        combat_encounter_basis = raw_observation.get("combat_encounter_basis")
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
            or not isinstance(scene_catalog_match, bool)
            or content_kind not in CANDIDATE_FRAME_CONTENT_KINDS
            or interface_kind not in CANDIDATE_INTERFACE_KINDS
            or not isinstance(prominent_event_portrait, bool)
            or not isinstance(cinematic_event_presentation, bool)
            or not isinstance(visible_dialogue_text, bool)
            or dialogue_text_presentation not in DIALOGUE_TEXT_PRESENTATIONS
            or not isinstance(visible_action, bool)
            or not isinstance(visible_character_or_enemy, bool)
            or combat_encounter_kind not in COMBAT_ENCOUNTER_KINDS
            or combat_encounter_basis not in COMBAT_ENCOUNTER_BASES
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
                    scene_catalog_match=scene_catalog_match,
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
                    combat_encounter_kind=cast(
                        CombatEncounterKind,
                        combat_encounter_kind,
                    ),
                    combat_encounter_basis=cast(
                        CombatEncounterBasis,
                        combat_encounter_basis,
                    ),
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


def _relationship_repair_error(error: VisionRuntimeError) -> VisionRuntimeError:
    """部分repair後の全体検証failureを専用codeへ正規化する。"""
    if (
        error.validation_code
        == "candidate_annotation_relationship_repair_response_empty"
    ):
        return _schema_error("candidate_annotation_relationship_repair_schema_invalid")
    if error.validation_code is not None and error.validation_code.startswith(
        "candidate_annotation_relationship_repair_"
    ):
        return error
    if error.reason in {
        VisionRuntimeFailureReason.SCHEMA_INVALID,
        VisionRuntimeFailureReason.DOMAIN_INVALID,
    }:
        return _domain_error("candidate_annotation_relationship_repair_domain_invalid")
    return error


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
        "candidate_annotation_relationship_repair_prompt_version": (
            CANDIDATE_ANNOTATION_RELATIONSHIP_REPAIR_PROMPT_VERSION
        ),
        "candidate_annotation_relationship_repair_schema_version": (
            CANDIDATE_ANNOTATION_RELATIONSHIP_REPAIR_SCHEMA_VERSION
        ),
        "candidate_annotation_relationship_repair_stage_contract_version": (
            CANDIDATE_ANNOTATION_RELATIONSHIP_REPAIR_STAGE_CONTRACT_VERSION
        ),
        "candidate_annotation_relationship_repair_num_predict": (
            CANDIDATE_ANNOTATION_RELATIONSHIP_REPAIR_NUM_PREDICT
        ),
        "candidate_annotation_relationship_repair_evidence_max_length": (
            CANDIDATE_ANNOTATION_RELATIONSHIP_REPAIR_EVIDENCE_MAX_LENGTH
        ),
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
            COMBAT_VISIBILITY_EDGE_AUDIT_SCHEMA_VERSION
        ),
        "combat_visibility_edge_audit_stage_contract_version": (
            COMBAT_VISIBILITY_EDGE_AUDIT_STAGE_CONTRACT_VERSION
        ),
        "combat_visibility_edge_strip_version": (COMBAT_VISIBILITY_EDGE_STRIP_VERSION),
        "publication_boundary_verification_prompt_version": (
            PUBLICATION_BOUNDARY_VERIFICATION_PROMPT_VERSION
        ),
        "publication_boundary_verification_schema_version": (
            PUBLICATION_BOUNDARY_VERIFICATION_SCHEMA_VERSION
        ),
        "publication_boundary_verification_stage_contract_version": (
            PUBLICATION_BOUNDARY_VERIFICATION_STAGE_CONTRACT_VERSION
        ),
        "cinematic_letterbox_detection_version": (
            CINEMATIC_LETTERBOX_DETECTION_VERSION
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
    """外周strip監査のcache identity入力を返す。"""
    return {
        "frame_candidate": {
            "id": candidate.identifier,
            "image_sha256": hashlib.sha256(candidate.image_bytes).hexdigest(),
        },
        "model": {**model.semantic_input(), "num_ctx": num_ctx},
        "generation_options": _semantic_generation_options(),
        "prompt_version": COMBAT_VISIBILITY_EDGE_AUDIT_PROMPT_VERSION,
        "schema_version": COMBAT_VISIBILITY_EDGE_AUDIT_SCHEMA_VERSION,
        "stage_contract_version": COMBAT_VISIBILITY_EDGE_AUDIT_STAGE_CONTRACT_VERSION,
        "edge_strip_version": COMBAT_VISIBILITY_EDGE_STRIP_VERSION,
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
            COMBAT_VISIBILITY_EDGE_AUDIT_SCHEMA_VERSION,
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
    observation: CombatVisibilityObservation,
) -> bool:
    """playerと敵本体が見え、敵が構図内に収まる場合だけ許可する。"""
    (
        player_body_visibility,
        opponent_body_visibility,
        opponent_body_framing,
        effect_only_frame,
    ) = observation
    return (
        player_body_visibility != "absent"
        and opponent_body_visibility == "clear"
        and opponent_body_framing == "complete"
        and not effect_only_frame
    )


def _is_consistent_noncombat_or_publishable_combat_visibility(
    first: CombatVisibilityObservation,
    confirmation: CombatVisibilityObservation,
) -> bool:
    """二回とも敵不在、または二回とも掲載可能な戦闘の場合だけ許可する。"""
    observations = (first, confirmation)
    opponent_is_observed = any(
        opponent_body_visibility != "absent" or opponent_body_framing != "absent"
        for (
            _player_body_visibility,
            opponent_body_visibility,
            opponent_body_framing,
            _effect_only_frame,
        ) in observations
    )
    if opponent_is_observed:
        return all(_is_publishable_combat_visibility(item) for item in observations)
    return all(
        not effect_only_frame
        for (
            _player_body_visibility,
            _opponent_body_visibility,
            _opponent_body_framing,
            effect_only_frame,
        ) in observations
    )


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
