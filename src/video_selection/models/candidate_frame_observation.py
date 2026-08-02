"""Candidate Moment内の一つのframeに対する構造化された意味観測。"""

from dataclasses import dataclass
from typing import Literal, cast, get_args

from .candidate_annotation import (
    EXPLANATION_VALUES,
    SCREEN_TEXT_KINDS,
    SPOILER_RISKS,
    BlogImageType,
    ExplanationValue,
    ScreenTextKind,
    SpoilerRisk,
)
from .combat_encounter_kind import COMBAT_ENCOUNTER_KINDS, CombatEncounterKind
from .frame_candidate import FrameCandidate
from .scene_catalog_entry import is_valid_scene_slug

CandidateFrameContentKind = Literal[
    "gameplay_action",
    "gameplay_idle",
    "event_dialogue",
    "event_action",
    "event_setup",
    "document",
    "shop",
    "map",
    "save",
    "tutorial_help",
    "other_interface",
    "title",
    "other",
]
CandidateInterfaceKind = Literal[
    "none",
    "document",
    "shop",
    "map",
    "save",
    "tutorial_help",
    "other_interface",
    "title",
]
PrimarySubjectVisibility = Literal["clear", "partial", "absent"]
TransientObstruction = Literal["none", "partial", "severe"]
CharacterBodyVisibility = Literal["clear", "partial", "absent"]
DialogueTextPresentation = Literal[
    "none",
    "dialogue_box",
    "speech_bubble",
    "subtitle_overlay",
    "other",
]

CANDIDATE_FRAME_CONTENT_KINDS = cast(
    tuple[CandidateFrameContentKind, ...],
    get_args(CandidateFrameContentKind),
)
CANDIDATE_INTERFACE_KINDS = cast(
    tuple[CandidateInterfaceKind, ...],
    get_args(CandidateInterfaceKind),
)
PRIMARY_SUBJECT_VISIBILITIES = cast(
    tuple[PrimarySubjectVisibility, ...],
    get_args(PrimarySubjectVisibility),
)
TRANSIENT_OBSTRUCTIONS = cast(
    tuple[TransientObstruction, ...],
    get_args(TransientObstruction),
)
CHARACTER_BODY_VISIBILITIES = cast(
    tuple[CharacterBodyVisibility, ...],
    get_args(CharacterBodyVisibility),
)
DIALOGUE_TEXT_PRESENTATIONS = cast(
    tuple[DialogueTextPresentation, ...],
    get_args(DialogueTextPresentation),
)

_BLOG_IMAGE_TYPES: dict[CandidateFrameContentKind, BlogImageType] = {
    "gameplay_action": "normal_gameplay",
    "gameplay_idle": "normal_gameplay",
    "event_dialogue": "event",
    "event_action": "event",
    "event_setup": "event",
    "document": "menu",
    "shop": "menu",
    "map": "menu",
    "save": "menu",
    "tutorial_help": "menu",
    "other_interface": "menu",
    "title": "title",
    "other": "other",
}
_NO_EXPLANATION_CONTENT = frozenset({"event_setup", "document", "tutorial_help"})
_MENU_TEXT_CONTENT = frozenset(
    {"document", "shop", "map", "save", "tutorial_help", "other_interface"}
)
_INTERFACE_CONTENT_KINDS: dict[CandidateInterfaceKind, CandidateFrameContentKind] = {
    "document": "document",
    "shop": "shop",
    "map": "map",
    "save": "save",
    "tutorial_help": "tutorial_help",
    "other_interface": "other_interface",
    "title": "title",
}


@dataclass(frozen=True)
class CandidateFrameObservation:
    """一つの入力frameへmodelが返したenumと決定的な正規化値を保持する。"""

    candidate: FrameCandidate
    scene_slug: str
    content_kind: CandidateFrameContentKind
    interface_kind: CandidateInterfaceKind
    prominent_event_portrait: bool
    cinematic_event_presentation: bool
    visible_dialogue_text: bool
    dialogue_text_presentation: DialogueTextPresentation
    visible_action: bool
    visible_character_or_enemy: bool
    combat_encounter_kind: CombatEncounterKind
    player_body_visibility: CharacterBodyVisibility
    opponent_body_visibility: CharacterBodyVisibility
    effect_only_frame: bool
    explanation_value: ExplanationValue
    screen_text_kind: ScreenTextKind
    primary_subject_visibility: PrimarySubjectVisibility
    transient_obstruction: TransientObstruction
    spoiler_risk: SpoilerRisk
    spoiler_evidence: str
    scene_catalog_match: bool = True

    def __post_init__(self) -> None:
        """frame identity、enum、Spoiler evidenceの関係を検証する。"""
        if (
            not self.candidate.image_bytes
            or not is_valid_scene_slug(self.scene_slug)
            or self.content_kind not in CANDIDATE_FRAME_CONTENT_KINDS
            or self.interface_kind not in CANDIDATE_INTERFACE_KINDS
            or not isinstance(self.prominent_event_portrait, bool)
            or not isinstance(self.cinematic_event_presentation, bool)
            or not isinstance(self.visible_dialogue_text, bool)
            or self.dialogue_text_presentation not in DIALOGUE_TEXT_PRESENTATIONS
            or self.visible_dialogue_text != (self.dialogue_text_presentation != "none")
            or not isinstance(self.visible_action, bool)
            or not isinstance(self.visible_character_or_enemy, bool)
            or self.combat_encounter_kind not in COMBAT_ENCOUNTER_KINDS
            or self.player_body_visibility not in CHARACTER_BODY_VISIBILITIES
            or self.opponent_body_visibility not in CHARACTER_BODY_VISIBILITIES
            or not isinstance(self.effect_only_frame, bool)
            or self.explanation_value not in EXPLANATION_VALUES
            or self.screen_text_kind not in SCREEN_TEXT_KINDS
            or self.primary_subject_visibility not in PRIMARY_SUBJECT_VISIBILITIES
            or self.transient_obstruction not in TRANSIENT_OBSTRUCTIONS
            or self.spoiler_risk not in SPOILER_RISKS
            or (self.spoiler_risk == "none") != (not self.spoiler_evidence)
            or not isinstance(self.scene_catalog_match, bool)
        ):
            msg = "Candidate Frame Observationのdomain fieldが不正です"
            raise ValueError(msg)

    @property
    def combat_action(self) -> bool:
        """画像内で戦闘が観測されたかを戦闘種別から返す。"""
        return self.combat_encounter_kind != "not_combat"

    @property
    def blog_image_type(self) -> BlogImageType:
        """視覚内容から決定的なBlog Image Typeを返す。"""
        return _BLOG_IMAGE_TYPES[self.effective_content_kind]

    @property
    def effective_content_kind(self) -> CandidateFrameContentKind:
        """単純な視覚観測を優先して曖昧なmodel分類を正規化する。"""
        if self.interface_kind not in {"none", "other_interface"}:
            return _INTERFACE_CONTENT_KINDS[self.interface_kind]
        if self.visible_dialogue_text and (
            self.content_kind == "event_dialogue"
            or self.prominent_event_portrait
            or self.cinematic_event_presentation
        ):
            return "event_dialogue"
        if self.interface_kind == "other_interface" and not self.visible_action:
            return "other_interface"
        if (
            (self.prominent_event_portrait or self.cinematic_event_presentation)
            and not self.visible_dialogue_text
            and not self.visible_action
        ):
            return "event_setup"
        if self.content_kind == "event_dialogue" and not self.visible_dialogue_text:
            return "event_action" if self.visible_action else "event_setup"
        if self.content_kind == "event_action" and not self.visible_action:
            return "event_setup"
        if self.content_kind == "gameplay_action" and not self.visible_action:
            return "gameplay_idle"
        return self.content_kind

    @property
    def effective_explanation_value(self) -> ExplanationValue:
        """ブログ掲載を妨げる観測へ説明価値なしを適用する。"""
        analysis = self.candidate.analysis
        is_low_information = (
            analysis is not None
            and analysis.quality_score < 0.35
            and analysis.metrics.information_score < 0.15
            and analysis.metrics.visibility_score < 0.85
        )
        if (
            self.effective_content_kind in _NO_EXPLANATION_CONTENT
            or self.effective_content_kind == "save"
            or (
                self.effective_content_kind == "shop"
                and not self.visible_character_or_enemy
            )
            or (self.combat_action and self.opponent_body_visibility != "clear")
            or self.effect_only_frame
            or self.primary_subject_visibility == "absent"
            or self.transient_obstruction == "severe"
            or is_low_information
        ):
            return "none"
        return self.explanation_value

    @property
    def effective_screen_text_kind(self) -> ScreenTextKind:
        """content kindと矛盾しない画面内text roleを返す。"""
        content_kind = self.effective_content_kind
        if content_kind in _MENU_TEXT_CONTENT:
            return "menu"
        if content_kind == "event_dialogue":
            return "dialogue"
        if content_kind == "event_setup":
            return "none"
        if content_kind == "title":
            return "title"
        return self.screen_text_kind
