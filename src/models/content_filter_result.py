"""content filter の出力."""

from dataclasses import dataclass

from .adaptive_scores import AdaptiveScores
from .analyzed_image import AnalyzedImage
from .whole_input_profile import WholeInputProfile


@dataclass(frozen=True)
class ContentFilterResult:
    """入力全体適応スコアと hard reject 結果."""

    kept_images: list[AnalyzedImage]
    adaptive_scores_by_path: dict[str, AdaptiveScores]
    rejected_by_content_filter: int
    content_filter_breakdown: dict[str, int]
    whole_input_profile: WholeInputProfile
