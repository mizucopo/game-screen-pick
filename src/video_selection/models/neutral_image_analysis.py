"""一つのnative frameのNeutral Image Analysis。"""

from dataclasses import dataclass

from .content_reject_reason import ContentRejectReason
from .neutral_image_metrics import NeutralImageMetrics


@dataclass(frozen=True)
class NeutralImageAnalysis:
    """model非依存のmetrics、特徴、採否を保持する。"""

    source_pts: int
    metrics: NeutralImageMetrics
    quality_score: float
    visual_feature: tuple[float, ...]
    grayscale_signature: bytes
    reject_reason: ContentRejectReason | None

    @property
    def eligible(self) -> bool:
        """Frame Candidateとして利用可能かを返す。"""
        return self.reject_reason is None
