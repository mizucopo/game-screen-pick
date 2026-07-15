"""Frame Candidate Extraction Stageのmetric。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class FrameCandidateExtractionMetrics:
    """抽出時間、Moment、reject、dedupe、proxy容量を保持する。"""

    wall_seconds: float
    cpu_seconds: float
    density_cap: int
    actual_moment_count: int
    native_frame_count: int
    reject_breakdown: dict[str, int]
    deduplicated_frame_count: int
    zero_frame_moment_count: int
    frame_candidate_count: int
    frame_candidate_bytes: int
