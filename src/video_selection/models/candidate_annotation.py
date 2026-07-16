"""一つのCandidate Momentの意味annotation。"""

import hashlib
from dataclasses import dataclass
from typing import Literal

from .frame_candidate import FrameCandidate

BlogImageType = Literal["normal_gameplay", "event", "menu", "title", "other"]
ExplanationValue = Literal["none", "low", "medium", "high"]
ScreenTextKind = Literal["none", "dialogue", "menu", "title", "hud", "other"]
ContextCueRelevance = Literal["unavailable", "none", "weak", "strong"]
SpoilerRisk = Literal["none", "low", "medium", "high"]


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
            or self.blog_image_type
            not in {"normal_gameplay", "event", "menu", "title", "other"}
            or self.explanation_value not in {"none", "low", "medium", "high"}
            or self.screen_text_kind
            not in {"none", "dialogue", "menu", "title", "hud", "other"}
            or self.context_relevance not in {"unavailable", "none", "weak", "strong"}
            or self.spoiler_risk not in {"none", "low", "medium", "high"}
            or len(self.supporting_context_cue_ids)
            != len(set(self.supporting_context_cue_ids))
            or self.context_relevance in {"unavailable", "none"}
            and self.supporting_context_cue_ids
            or self.context_relevance in {"weak", "strong"}
            and not self.supporting_context_cue_ids
            or self.spoiler_risk == "none"
            and self.spoiler_evidence
            or self.spoiler_risk != "none"
            and not self.spoiler_evidence.strip()
        ):
            msg = "Candidate Annotationのdomain fieldが不正です"
            raise ValueError(msg)
