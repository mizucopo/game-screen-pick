"""Video Source内のCandidate Moment。"""

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

MomentEvidence = Literal["heartbeat", "scene"]


@dataclass(frozen=True)
class CandidateMoment:
    """exact anchor、検出根拠、refinement結果を保持する。"""

    identifier: str
    source_pts: int
    anchor_time: Fraction
    timeline_segment_id: str
    evidence: tuple[MomentEvidence, ...]
    proxy_quality_score: float
    frame_candidate_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """ID、時刻、根拠、画質を検証する。"""
        if (
            not self.identifier.startswith("mom_")
            or len(self.identifier) != 68
            or any(
                character not in "0123456789abcdef" for character in self.identifier[4:]
            )
        ):
            msg = "Candidate Moment IDにはmom_と64桁SHA-256が必要です"
            raise ValueError(msg)
        if self.anchor_time < 0 or not self.evidence:
            msg = "Candidate Momentには時刻と検出根拠が必要です"
            raise ValueError(msg)
        if tuple(sorted(set(self.evidence))) != self.evidence:
            msg = "Candidate Moment evidenceは重複のない順序付き値が必要です"
            raise ValueError(msg)
        if not math.isfinite(self.proxy_quality_score):
            msg = "Candidate Momentには有限のproxy画質値が必要です"
            raise ValueError(msg)
