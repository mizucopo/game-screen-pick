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
from .frame_candidate import FrameCandidate
from .scene_catalog_entry import is_valid_scene_slug

CandidateFrameContentKind = Literal[
    "gameplay_action",
    "gameplay_idle",
    "event_dialogue",
    "event_action",
    "event_setup",
    "shop",
    "map",
    "save",
    "tutorial_help",
    "other_interface",
    "title",
    "other",
]
PrimarySubjectVisibility = Literal["clear", "partial", "absent"]
TransientObstruction = Literal["none", "partial", "severe"]

CANDIDATE_FRAME_CONTENT_KINDS = cast(
    tuple[CandidateFrameContentKind, ...],
    get_args(CandidateFrameContentKind),
)
PRIMARY_SUBJECT_VISIBILITIES = cast(
    tuple[PrimarySubjectVisibility, ...],
    get_args(PrimarySubjectVisibility),
)
TRANSIENT_OBSTRUCTIONS = cast(
    tuple[TransientObstruction, ...],
    get_args(TransientObstruction),
)

_BLOG_IMAGE_TYPES: dict[CandidateFrameContentKind, BlogImageType] = {
    "gameplay_action": "normal_gameplay",
    "gameplay_idle": "normal_gameplay",
    "event_dialogue": "event",
    "event_action": "event",
    "event_setup": "event",
    "shop": "menu",
    "map": "menu",
    "save": "menu",
    "tutorial_help": "menu",
    "other_interface": "menu",
    "title": "title",
    "other": "other",
}
_NO_EXPLANATION_CONTENT = frozenset({"event_setup", "tutorial_help"})
_MENU_TEXT_CONTENT = frozenset(
    {"shop", "map", "save", "tutorial_help", "other_interface"}
)


@dataclass(frozen=True)
class CandidateFrameObservation:
    """一つの入力frameへmodelが返したenumと決定的な正規化値を保持する。"""

    candidate: FrameCandidate
    scene_slug: str
    content_kind: CandidateFrameContentKind
    explanation_value: ExplanationValue
    screen_text_kind: ScreenTextKind
    primary_subject_visibility: PrimarySubjectVisibility
    transient_obstruction: TransientObstruction
    spoiler_risk: SpoilerRisk
    spoiler_evidence: str

    def __post_init__(self) -> None:
        """frame identity、enum、Spoiler evidenceの関係を検証する。"""
        if (
            not self.candidate.image_bytes
            or not is_valid_scene_slug(self.scene_slug)
            or self.content_kind not in CANDIDATE_FRAME_CONTENT_KINDS
            or self.explanation_value not in EXPLANATION_VALUES
            or self.screen_text_kind not in SCREEN_TEXT_KINDS
            or self.primary_subject_visibility not in PRIMARY_SUBJECT_VISIBILITIES
            or self.transient_obstruction not in TRANSIENT_OBSTRUCTIONS
            or self.spoiler_risk not in SPOILER_RISKS
            or (self.spoiler_risk == "none") != (not self.spoiler_evidence)
        ):
            msg = "Candidate Frame Observationのdomain fieldが不正です"
            raise ValueError(msg)

    @property
    def blog_image_type(self) -> BlogImageType:
        """視覚内容から決定的なBlog Image Typeを返す。"""
        return _BLOG_IMAGE_TYPES[self.content_kind]

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
            self.content_kind in _NO_EXPLANATION_CONTENT
            or self.primary_subject_visibility == "absent"
            or self.transient_obstruction == "severe"
            or is_low_information
        ):
            return "none"
        return self.explanation_value

    @property
    def effective_screen_text_kind(self) -> ScreenTextKind:
        """content kindと矛盾しない画面内text roleを返す。"""
        if self.content_kind in _MENU_TEXT_CONTENT:
            return "menu"
        if self.content_kind == "event_dialogue":
            return "dialogue"
        if self.content_kind == "event_setup":
            return "none"
        if self.content_kind == "title":
            return "title"
        return self.screen_text_kind
