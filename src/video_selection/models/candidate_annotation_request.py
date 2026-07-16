"""一つのCandidate Momentのannotation semantic入力。"""

import re
from dataclasses import dataclass
from fractions import Fraction

from .candidate_moment import CandidateMoment
from .context_cue import ContextCue
from .frame_candidate import FrameCandidate

_POLICY_VERSION_PATTERN = re.compile(r"[0-9A-Za-z][0-9A-Za-z._/-]{0,127}")


@dataclass(frozen=True)
class CandidateAnnotationRequest:
    """Moment、1〜3 frame、近傍Cue、進行位置、意図を保持する。"""

    moment: CandidateMoment
    frame_candidates: tuple[FrameCandidate, ...]
    context_cues: tuple[ContextCue, ...]
    video_set_progress: Fraction
    selection_intent: str
    cue_selection_policy_version: str

    def __post_init__(self) -> None:
        """Candidate Momentに属する一意な入力だけを受理する。"""
        frame_ids = tuple(item.identifier for item in self.frame_candidates)
        cue_ids = tuple(item.identifier for item in self.context_cues)
        frame_video_ids = {
            item.video_fingerprint
            for item in self.frame_candidates
            if item.video_fingerprint is not None
        }
        cue_video_ids = {
            item.video_fingerprint
            for item in self.context_cues
            if item.video_fingerprint
        }
        if (
            not 1 <= len(self.frame_candidates) <= 3
            or frame_ids != self.moment.frame_candidate_ids
            or len(frame_ids) != len(set(frame_ids))
            or any(not item.image_bytes for item in self.frame_candidates)
            or len(cue_ids) != len(set(cue_ids))
            or not 0 <= self.video_set_progress < 1
            or not self.selection_intent.strip()
            or _POLICY_VERSION_PATTERN.fullmatch(self.cue_selection_policy_version)
            is None
            or len(frame_video_ids) > 1
            or len(cue_video_ids) > 1
            or frame_video_ids
            and cue_video_ids
            and frame_video_ids != cue_video_ids
        ):
            msg = "Candidate Annotation requestが不正です"
            raise ValueError(msg)
