"""入力全体適応の hard reject フィルタ."""

import numpy as np

from ..constants.content_filter_thresholds import (
    TEMPORAL_SIMILARITY_THRESHOLD,
    TEMPORAL_VISIBILITY_MARGIN,
)
from ..models.adaptive_scores import AdaptiveScores
from ..models.analyzed_image import AnalyzedImage
from ..models.content_filter_result import ContentFilterResult
from ..utils.vector_utils import VectorUtils
from .static_reject_classifier import StaticRejectClassifier
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

    def __init__(self, profiler: WholeInputProfiler):
        """ContentFilterを初期化する."""
        self.profiler = profiler
        self._static_classifier = StaticRejectClassifier()

    def filter(self, images: list[AnalyzedImage]) -> ContentFilterResult:
        """入力全体の分布を使って hard reject を適用する."""
        profile = self.profiler.build_profile(images)
        adaptive_scores = self.profiler.score_images(images)
        breakdown = dict.fromkeys(self.REJECTION_REASONS, 0)
        rejected_reasons_by_index = {
            index: self._static_classifier.classify(
                image=image,
                profile=profile,
                adaptive_scores=adaptive_scores[image.path],
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
            adaptive_scores_by_path=adaptive_scores,
            rejected_by_content_filter=sum(breakdown.values()),
            content_filter_breakdown=breakdown,
            whole_input_profile=profile,
        )

    def _find_temporal_rejections(
        self,
        images: list[AnalyzedImage],
        adaptive_scores: dict[str, AdaptiveScores],
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
            if prev_next_similarity < TEMPORAL_SIMILARITY_THRESHOLD:
                continue

            current_visibility = adaptive_scores[images[index].path].visibility_score
            prev_visibility = adaptive_scores[images[index - 1].path].visibility_score
            next_visibility = adaptive_scores[images[index + 1].path].visibility_score
            if current_visibility + TEMPORAL_VISIBILITY_MARGIN < min(
                prev_visibility, next_visibility
            ):
                rejected_indices.add(index)

        return rejected_indices
