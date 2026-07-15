"""一つのVideo SourceのFrame Candidate抽出結果。"""

from dataclasses import dataclass

from .candidate_moment import CandidateMoment
from .content_reject_reason import ContentRejectReason
from .frame_candidate import FrameCandidate


@dataclass(frozen=True)
class FrameCandidateExtraction:
    """Moment、共有Frame Candidate、抽出診断を保持する。"""

    moments: tuple[CandidateMoment, ...]
    candidates: tuple[FrameCandidate, ...]
    native_frame_count: int
    reject_breakdown: dict[str, int]
    deduplicated_frame_count: int
    zero_frame_moment_count: int

    def __post_init__(self) -> None:
        """件数とstable reason breakdownを検証する。"""
        if set(self.reject_breakdown) != {
            reason.value for reason in ContentRejectReason
        }:
            msg = "Frame Candidate抽出には全stable reject reasonが必要です"
            raise ValueError(msg)
        if any(
            value < 0
            for value in (
                self.native_frame_count,
                self.deduplicated_frame_count,
                self.zero_frame_moment_count,
                *self.reject_breakdown.values(),
            )
        ):
            msg = "Frame Candidate抽出の診断件数は0以上である必要があります"
            raise ValueError(msg)
