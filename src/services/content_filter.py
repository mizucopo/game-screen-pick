"""入力全体適応の hard reject フィルタ."""

import numpy as np

from ..models.adaptive_scores import AdaptiveScores
from ..models.analyzed_image import AnalyzedImage
from ..models.content_filter_result import ContentFilterResult
from ..models.whole_input_profile import WholeInputProfile
from ..utils.vector_utils import VectorUtils
from .whole_input_profiler import WholeInputProfiler


class ContentFilter:
    """真っ暗・真っ白・単色・暗転途中を hard reject する."""

    REJECTION_REASONS = (
        "blackout",
        "whiteout",
        "single_tone",
        "fade_transition",
        "temporal_transition",
    )
    TEMPORAL_SIMILARITY_THRESHOLD = 0.90
    TEMPORAL_VISIBILITY_MARGIN = 0.25

    def __init__(self, profiler: WholeInputProfiler):
        """ContentFilterを初期化する."""
        self.profiler = profiler

    def filter(self, images: list[AnalyzedImage]) -> ContentFilterResult:
        """入力全体の分布を使って hard reject を適用する."""
        profile = self.profiler.build_profile(images)
        adaptive_scores = self.profiler.score_images(images)
        breakdown = dict.fromkeys(self.REJECTION_REASONS, 0)
        rejected_reasons_by_index = {
            index: self._classify_static_rejection_reason(
                image=image,
                profile=profile,
                adaptive_scores=adaptive_scores[id(image)],
            )
            for index, image in enumerate(images)
        }
        temporal_rejected_indices = self._find_temporal_rejections(
            images,
            adaptive_scores,
            rejected_reasons_by_index,
        )
        kept_images: list[AnalyzedImage] = []

        for index, image in enumerate(images):
            reason = rejected_reasons_by_index[index]
            if reason is None and index in temporal_rejected_indices:
                reason = "temporal_transition"
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
    def _classify_static_rejection_reason(
        image: AnalyzedImage,
        profile: WholeInputProfile,
        adaptive_scores: AdaptiveScores,
    ) -> str | None:
        """固定条件に従って hard reject 理由を返す."""
        raw = image.raw_metrics
        p10_range = max(12.0, profile.luminance_range.p10)
        p25_range = max(14.0, profile.luminance_range.p25)

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
            (raw.near_black_ratio >= 0.65 or raw.near_white_ratio >= 0.65)
            and adaptive_scores.visibility_score < 0.30
            and (
                raw.luminance_range <= p25_range
                or adaptive_scores.information_score < 0.35
            )
        ):
            return "fade_transition"

        return None

    def _find_temporal_rejections(
        self,
        images: list[AnalyzedImage],
        adaptive_scores: dict[int, AdaptiveScores],
        static_reasons_by_index: dict[int, str | None],
    ) -> set[int]:
        """前後関係から暗転途中フレームを検出する."""
        if len(images) < 3:
            return set()

        normalized_features = [
            VectorUtils.safe_l2_normalize(
                np.asarray(image.content_features, dtype=np.float32)
            )
            for image in images
        ]
        rejected_indices: set[int] = set()

        for index in range(1, len(images) - 1):
            if static_reasons_by_index[index] is not None:
                continue
            if static_reasons_by_index[index - 1] is not None:
                continue
            if static_reasons_by_index[index + 1] is not None:
                continue

            prev_feature = normalized_features[index - 1]
            next_feature = normalized_features[index + 1]
            prev_next_similarity = float(
                np.clip(prev_feature @ next_feature, -1.0, 1.0)
            )
            if prev_next_similarity < self.TEMPORAL_SIMILARITY_THRESHOLD:
                continue

            current_visibility = adaptive_scores[id(images[index])].visibility_score
            prev_visibility = adaptive_scores[id(images[index - 1])].visibility_score
            next_visibility = adaptive_scores[id(images[index + 1])].visibility_score
            if current_visibility + self.TEMPORAL_VISIBILITY_MARGIN < min(
                prev_visibility, next_visibility
            ):
                rejected_indices.add(index)

        return rejected_indices
