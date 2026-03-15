"""content filter の出力."""

from dataclasses import dataclass

from .adaptive_scores import AdaptiveScores
from .analyzed_image import AnalyzedImage


@dataclass(frozen=True)
class ContentFilterResult:
    """入力全体適応スコアと hard reject 結果."""

    kept_images: list[AnalyzedImage]
    adaptive_scores_by_image_id: dict[int, AdaptiveScores]
    rejected_by_content_filter: int
    content_filter_breakdown: dict[str, int]
