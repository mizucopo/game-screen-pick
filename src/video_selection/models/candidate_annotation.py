"""一つのCandidate Momentの意味annotation。"""

import hashlib
import unicodedata
from dataclasses import dataclass
from typing import Literal, cast, get_args

from .combat_encounter_basis import (
    COMBAT_ENCOUNTER_BASES,
    ORDINARY_COMBAT_ENCOUNTER_BASES,
    CombatEncounterBasis,
    combat_encounter_classification_is_valid,
)
from .combat_encounter_kind import COMBAT_ENCOUNTER_KINDS, CombatEncounterKind
from .frame_candidate import FrameCandidate
from .report_value import string_looks_private

BlogImageType = Literal["normal_gameplay", "event", "menu", "title", "other"]
ExplanationValue = Literal["none", "low", "medium", "high"]
ScreenTextKind = Literal["none", "dialogue", "menu", "title", "hud", "other"]
ContextCueRelevance = Literal["unavailable", "none", "weak", "strong"]
SpoilerRisk = Literal["none", "low", "medium", "high"]
SelectionCoverageFacet = Literal["ordinary_combat", "event"]

BLOG_IMAGE_TYPES = cast(tuple[BlogImageType, ...], get_args(BlogImageType))
EXPLANATION_VALUES = cast(tuple[ExplanationValue, ...], get_args(ExplanationValue))
SCREEN_TEXT_KINDS = cast(tuple[ScreenTextKind, ...], get_args(ScreenTextKind))
CONTEXT_CUE_RELEVANCES = cast(
    tuple[ContextCueRelevance, ...],
    get_args(ContextCueRelevance),
)
SPOILER_RISKS = cast(tuple[SpoilerRisk, ...], get_args(SpoilerRisk))
SELECTION_COVERAGE_FACETS = cast(
    tuple[SelectionCoverageFacet, ...],
    get_args(SelectionCoverageFacet),
)

_MIN_VERBATIM_SPAN_LENGTH = 6
_MIN_VERBATIM_CUE_LENGTH = 3


def candidate_annotation_relationships_are_valid(
    context_relevance: ContextCueRelevance,
    supporting_context_cue_ids: tuple[str, ...],
    spoiler_risk: SpoilerRisk,
    spoiler_evidence: str,
) -> bool:
    """Annotation内のCue参照とSpoiler evidenceの相関を検証する。"""
    if len(supporting_context_cue_ids) != len(set(supporting_context_cue_ids)):
        return False
    if bool(supporting_context_cue_ids) != (context_relevance in {"weak", "strong"}):
        return False
    if spoiler_risk == "none":
        return not spoiler_evidence
    return bool(spoiler_evidence.strip())


def candidate_annotation_context_is_valid(
    context_relevance: ContextCueRelevance,
    supporting_context_cue_ids: tuple[str, ...],
    available_context_cue_ids: tuple[str, ...],
) -> bool:
    """Cueの有無、relevance、参照IDの所属が整合することを検証する。"""
    if not set(supporting_context_cue_ids).issubset(available_context_cue_ids):
        return False
    if not available_context_cue_ids:
        return context_relevance == "unavailable" and not supporting_context_cue_ids
    return context_relevance != "unavailable"


def candidate_annotation_free_text_is_safe(
    annotation_texts: tuple[str, ...],
    raw_context_texts: tuple[str, ...],
) -> bool:
    """公開・cache対象の自由文にContext Cue本文が逐語再出力されないことを検証する。"""
    if any(string_looks_private(item) for item in annotation_texts):
        return False
    normalized_annotations = tuple(
        _normalize_verbatim_text(item) for item in annotation_texts
    )
    normalized_cues = tuple(
        normalized
        for item in raw_context_texts
        if len(normalized := _normalize_verbatim_text(item)) >= _MIN_VERBATIM_CUE_LENGTH
    )
    for cue in normalized_cues:
        span_length = min(len(cue), _MIN_VERBATIM_SPAN_LENGTH)
        cue_spans = {
            cue[index : index + span_length]
            for index in range(len(cue) - span_length + 1)
        }
        if any(
            span in annotation
            for annotation in normalized_annotations
            for span in cue_spans
        ):
            return False
    return True


def privacy_safe_candidate_text(
    generated: str,
    fallback: str,
    raw_context_texts: tuple[str, ...],
) -> tuple[str, bool]:
    """生成文を1行化し、非公開形式とContext Cueをfield単位で安全化する。"""
    normalized_generated = _single_line_text(generated)
    normalized_fallback = _single_line_text(fallback)
    if candidate_annotation_free_text_is_safe(
        (normalized_generated,),
        raw_context_texts,
    ):
        return normalized_generated, False
    if normalized_fallback and candidate_annotation_free_text_is_safe(
        (normalized_fallback,), raw_context_texts
    ):
        return normalized_fallback, True
    return "［…］", True


def _single_line_text(value: str) -> str:
    """公開自由文の空白を内容順を保った一行へ正規化する。"""
    return " ".join(value.split())


def _normalize_verbatim_text(value: str) -> str:
    """Unicode、空白、句読点の表記差を除いて逐語一致を比較可能にする。"""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(
        character
        for character in normalized
        if unicodedata.category(character)[0] in {"L", "N"}
    )


@dataclass(frozen=True)
class CandidateAnnotation:
    """Representative Frameと最終選定前の意味情報を保持する。"""

    candidate: FrameCandidate
    summary: str
    candidate_moment_id: str | None = None
    scene_slug: str = "other"
    blog_image_type: BlogImageType = "other"
    explanation_value: ExplanationValue = "none"
    frame_choice_reason: str | None = None
    screen_text_kind: ScreenTextKind = "none"
    context_relevance: ContextCueRelevance = "unavailable"
    supporting_context_cue_ids: tuple[str, ...] = ()
    spoiler_risk: SpoilerRisk = "none"
    spoiler_evidence: str = ""
    combat_encounter_kind: CombatEncounterKind = "not_combat"
    combat_encounter_basis: CombatEncounterBasis = "none"

    def __post_init__(self) -> None:
        """domain enum、所属ID、evidenceの整合を検証する。"""
        if self.candidate_moment_id is None:
            digest = hashlib.sha256(self.candidate.identifier.encode()).hexdigest()
            object.__setattr__(self, "candidate_moment_id", f"mom_{digest}")
        if self.frame_choice_reason is None:
            object.__setattr__(self, "frame_choice_reason", self.summary)
        moment_id = self.candidate_moment_id
        if (
            moment_id is None
            or not moment_id.startswith("mom_")
            or len(moment_id) != 68
            or any(character not in "0123456789abcdef" for character in moment_id[4:])
            or not self.summary.strip()
            or not self.scene_slug.strip()
            or not self.frame_choice_reason
            or not self.frame_choice_reason.strip()
            or self.blog_image_type not in BLOG_IMAGE_TYPES
            or self.explanation_value not in EXPLANATION_VALUES
            or self.screen_text_kind not in SCREEN_TEXT_KINDS
            or self.context_relevance not in CONTEXT_CUE_RELEVANCES
            or self.spoiler_risk not in SPOILER_RISKS
            or self.combat_encounter_kind not in COMBAT_ENCOUNTER_KINDS
            or self.combat_encounter_basis not in COMBAT_ENCOUNTER_BASES
            or not combat_encounter_classification_is_valid(
                self.combat_encounter_kind,
                self.combat_encounter_basis,
            )
            or not candidate_annotation_relationships_are_valid(
                self.context_relevance,
                self.supporting_context_cue_ids,
                self.spoiler_risk,
                self.spoiler_evidence,
            )
        ):
            msg = "Candidate Annotationのdomain fieldが不正です"
            raise ValueError(msg)

    @property
    def combat_action(self) -> bool:
        """画像内で戦闘が観測されたかを戦闘種別から返す。"""
        return self.combat_encounter_kind != "not_combat"

    @property
    def selection_coverage_facet(self) -> SelectionCoverageFacet | None:
        """条件付き最低coverageに使う画像内根拠のある役割を返す。"""
        if self.blog_image_type == "event":
            return "event"
        if (
            self.blog_image_type == "normal_gameplay"
            and self.combat_encounter_kind == "ordinary"
            and self.combat_encounter_basis in ORDINARY_COMBAT_ENCOUNTER_BASES
        ):
            return "ordinary_combat"
        return None
