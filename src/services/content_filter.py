"""入力全体適応の hard reject フィルタ."""

from ..models.analyzed_image import AnalyzedImage
from ..models.content_filter_result import ContentFilterResult
from ..models.whole_input_profile import WholeInputProfile
from .whole_input_profiler import WholeInputProfiler


class ContentFilter:
    """真っ暗・真っ白・単色・暗転途中を hard reject する."""

    REJECTION_REASONS = (
        "blackout",
        "whiteout",
        "single_tone",
        "fade_transition",
    )

    def __init__(self, profiler: WholeInputProfiler):
        """ContentFilterを初期化する."""
        self.profiler = profiler

    def filter(self, images: list[AnalyzedImage]) -> ContentFilterResult:
        """入力全体の分布を使って hard reject を適用する."""
        profile = self.profiler.build_profile(images)
        adaptive_scores = self.profiler.score_images(images)
        breakdown = dict.fromkeys(self.REJECTION_REASONS, 0)
        kept_images: list[AnalyzedImage] = []

        for image in images:
            reason = self._classify_rejection_reason(
                image=image,
                profile=profile,
                information_score=adaptive_scores[id(image)].information_score,
                distinctiveness_score=adaptive_scores[id(image)].distinctiveness_score,
            )
            if reason is None:
                kept_images.append(image)
                continue
            breakdown[reason] += 1

        return ContentFilterResult(
            kept_images=kept_images,
            adaptive_scores_by_image_id=adaptive_scores,
            rejected_by_content_filter=sum(breakdown.values()),
            content_filter_breakdown=breakdown,
        )

    @staticmethod
    def _classify_rejection_reason(
        image: AnalyzedImage,
        profile: WholeInputProfile,
        information_score: float,
        distinctiveness_score: float,
    ) -> str | None:
        """固定条件に従って hard reject 理由を返す."""
        raw = image.raw_metrics
        p10_range = max(12.0, profile.luminance_range.p10)

        if (
            raw.near_black_ratio >= 0.97
            and raw.luminance_entropy <= 0.35
            and raw.luminance_range <= p10_range
        ):
            return "blackout"

        if (
            raw.near_white_ratio >= 0.97
            and raw.luminance_entropy <= 0.35
            and raw.luminance_range <= p10_range
        ):
            return "whiteout"

        if (
            raw.dominant_tone_ratio >= 0.92
            and raw.luminance_range <= p10_range
            and raw.contrast <= profile.contrast.p25
            and raw.edge_density <= profile.edge_density.p25
        ):
            return "single_tone"

        if (
            (raw.near_black_ratio >= 0.80 or raw.near_white_ratio >= 0.80)
            and information_score < 0.20
            and distinctiveness_score < 0.15
        ):
            return "fade_transition"

        return None
