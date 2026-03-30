"""入力全体適応の hard reject フィルタ."""

import numpy as np

from ..constants.content_filter_thresholds import ContentFilterThresholds
from ..models.adaptive_scores import AdaptiveScores
from ..models.analyzed_image import AnalyzedImage
from ..models.content_filter_result import ContentFilterResult
from ..models.raw_metrics import RawMetrics
from ..models.whole_input_profile import WholeInputProfile
from ..utils.transition_metrics import (
    calculate_bright_washout_score,
    calculate_relative_transition_scores,
    calculate_system_ui_signal,
    calculate_veiled_transition_score,
)
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
            whole_input_profile=profile,
        )

    @staticmethod
    def _detect_blackout(raw: RawMetrics, p10_range: float) -> str | None:
        """ブラックアウトを検出する."""
        t = ContentFilterThresholds
        if (
            raw.near_black_ratio >= t.BLACKOUT_NEAR_BLACK_RATIO
            and raw.luminance_entropy <= t.BLACKOUT_LUMINANCE_ENTROPY_MAX
            and raw.luminance_range <= p10_range
        ):
            return "blackout"
        return None

    @staticmethod
    def _detect_whiteout(
        raw: RawMetrics,
        profile: WholeInputProfile,
        adaptive_scores: AdaptiveScores,
        bright_washout_score: float,
        p25_range: float,
    ) -> str | None:
        """ホワイトアウトを検出する（3条件）."""
        t = ContentFilterThresholds

        if (
            raw.near_white_ratio >= t.WHITEOUT_NEAR_WHITE_THRESHOLD
            and raw.luminance_entropy <= t.WHITEOUT_LUMINANCE_ENTROPY_THRESHOLD
            and raw.luminance_range <= p25_range
        ):
            return "whiteout"

        if (
            raw.brightness >= t.WHITEOUT_BRIGHTNESS_THRESHOLD
            and raw.contrast <= max(t.WHITEOUT_MAX_CONTRAST, profile.contrast.p10)
            and raw.edge_density <= max(t.WHITEOUT_MAX_EDGE_DENSITY, profile.edge_density.p10)
            and raw.dominant_tone_ratio >= t.WHITEOUT_MIN_DOMINANT_TONE_RATIO
        ):
            return "whiteout"

        if (
            bright_washout_score >= t.WHITEOUT_BRIGHT_WASHOUT_THRESHOLD
            and raw.brightness >= t.WHITEOUT_RELAXED_MIN_BRIGHTNESS
            and raw.contrast <= max(t.WHITEOUT_RELAXED_MAX_CONTRAST, profile.contrast.p25)
            and raw.edge_density <= max(t.WHITEOUT_RELAXED_MAX_EDGE_DENSITY, profile.edge_density.p25)
            and adaptive_scores.visibility_score <= t.WHITEOUT_RELAXED_MAX_VISIBILITY
        ):
            return "whiteout"

        return None

    @staticmethod
    def _detect_single_tone(
        raw: RawMetrics,
        profile: WholeInputProfile,
        p10_range: float,
    ) -> str | None:
        """単色画像を検出する."""
        t = ContentFilterThresholds
        if (
            raw.dominant_tone_ratio >= t.SINGLE_TONE_DOMINANT_RATIO
            and raw.luminance_range <= p10_range
            and raw.contrast <= profile.contrast.p25
            and raw.edge_density <= profile.edge_density.p25
        ):
            return "single_tone"
        return None

    @staticmethod
    def _detect_bright_transition(
        raw: RawMetrics,
        profile: WholeInputProfile,
        adaptive_scores: AdaptiveScores,
        bright_washout_score: float,
        relative_bright_transition_score: float,
    ) -> str | None:
        """相対的な明転遷移を検出する."""
        t = ContentFilterThresholds
        if (
            relative_bright_transition_score >= t.RELATIVE_BRIGHT_TRANSITION_THRESHOLD
            and (
                adaptive_scores.visibility_score <= t.RELATIVE_BRIGHT_VISIBILITY_MAX
                or adaptive_scores.information_score <= t.RELATIVE_BRIGHT_INFORMATION_MAX
            )
            and (
                raw.near_white_ratio >= profile.near_white_ratio.p90
                or bright_washout_score >= t.RELATIVE_BRIGHT_WASHOUT_MIN
            )
        ):
            if (
                relative_bright_transition_score >= t.RELATIVE_BRIGHT_EXTREME_TRANSITION
                or raw.near_white_ratio >= t.RELATIVE_BRIGHT_EXTREME_THRESHOLD
            ):
                return "whiteout"
            return "fade_transition"
        return None

    @staticmethod
    def _detect_dark_transition(
        raw: RawMetrics,
        adaptive_scores: AdaptiveScores,
        profile: WholeInputProfile,
        relative_dark_transition_score: float,
    ) -> str | None:
        """相対的な暗転遷移を検出する."""
        t = ContentFilterThresholds
        if (
            relative_dark_transition_score >= t.RELATIVE_DARK_TRANSITION_THRESHOLD
            and adaptive_scores.visibility_score <= t.RELATIVE_DARK_VISIBILITY_MAX
            and raw.brightness <= profile.brightness.p25
        ):
            return "fade_transition"
        return None

    @staticmethod
    def _classify_static_rejection_reason(
        image: AnalyzedImage,
        profile: WholeInputProfile,
        adaptive_scores: AdaptiveScores,
    ) -> str | None:
        """固定条件に従って hard reject 理由を返す."""
        raw = image.raw_metrics
        t = ContentFilterThresholds
        p10_range = max(t.LUMINANCE_RANGE_P10_MIN, profile.luminance_range.p10)
        p25_range = max(t.LUMINANCE_RANGE_P25_MIN, profile.luminance_range.p25)
        bright_washout_score = calculate_bright_washout_score(raw)
        system_ui_signal = calculate_system_ui_signal(image.layout_heuristics)
        veiled_transition_score = calculate_veiled_transition_score(
            raw,
            adaptive_scores,
            image.layout_heuristics,
            image.normalized_metrics,
        )
        (
            relative_bright_transition_score,
            relative_dark_transition_score,
            _relative_transition_score,
            _relative_transition_polarity,
        ) = calculate_relative_transition_scores(
            raw,
            profile,
        )

        if (reason := ContentFilter._detect_blackout(raw, p10_range)) is not None:
            return reason
        if (reason := ContentFilter._detect_whiteout(raw, profile, adaptive_scores, bright_washout_score, p25_range)) is not None:
            return reason
        if (reason := ContentFilter._detect_single_tone(raw, profile, p10_range)) is not None:
            return reason
        if (reason := ContentFilter._detect_bright_transition(raw, profile, adaptive_scores, bright_washout_score, relative_bright_transition_score)) is not None:
            return reason
        if (reason := ContentFilter._detect_dark_transition(raw, adaptive_scores, profile, relative_dark_transition_score)) is not None:
            return reason
        if ContentFilter._is_fade_transition(
            raw,
            profile,
            adaptive_scores,
            p25_range,
            bright_washout_score,
            veiled_transition_score,
            system_ui_signal,
        ):
            return "fade_transition"

        return None

    @classmethod
    def _is_fade_transition(
        cls,
        raw: RawMetrics,
        profile: WholeInputProfile,
        adaptive_scores: AdaptiveScores,
        p25_range: float,
        bright_washout_score: float,
        veiled_transition_score: float,
        system_ui_signal: float,
    ) -> bool:
        """暗転・明転・露出過多/不足の遷移フレームかどうかを返す."""
        obvious_fade = (
            (raw.near_black_ratio >= ContentFilterThresholds.OBVIOUS_FADE_EXTREME_RATIO or raw.near_white_ratio >= ContentFilterThresholds.OBVIOUS_FADE_EXTREME_RATIO)
            and adaptive_scores.visibility_score < ContentFilterThresholds.FADE_VISIBILITY_THRESHOLD
            and (
                raw.luminance_range <= p25_range
                or adaptive_scores.information_score < ContentFilterThresholds.FADE_INFORMATION_THRESHOLD
            )
        )
        if obvious_fade:
            return True

        bright_washout_fade = (
            bright_washout_score >= ContentFilterThresholds.BRIGHT_WASHOUT_FADE_THRESHOLD
            and (
                raw.brightness >= ContentFilterThresholds.BRIGHT_WASHOUT_FADE_MIN_BRIGHTNESS
                or raw.near_white_ratio >= ContentFilterThresholds.BRIGHT_WASHOUT_FADE_MIN_NEAR_WHITE_RATIO
            )
            and (
                adaptive_scores.visibility_score
                <= ContentFilterThresholds.BRIGHT_WASHOUT_FADE_MAX_VISIBILITY
                or adaptive_scores.information_score
                <= ContentFilterThresholds.BRIGHT_WASHOUT_FADE_MAX_INFORMATION
            )
        )
        if bright_washout_fade:
            return True

        veiled_fade = veiled_transition_score >= ContentFilterThresholds.VEILED_FADE_THRESHOLD and (
            bright_washout_score >= ContentFilterThresholds.VEILED_FADE_MIN_BRIGHT_WASHOUT
            or raw.near_white_ratio >= ContentFilterThresholds.VEILED_FADE_MIN_EXTREME_RATIO
            or raw.near_black_ratio >= ContentFilterThresholds.VEILED_FADE_MIN_EXTREME_RATIO
            or system_ui_signal >= ContentFilterThresholds.VEILED_FADE_MIN_SYSTEM_UI
        )
        if veiled_fade:
            return True

        bright_fade = (
            raw.near_white_ratio >= ContentFilterThresholds.DIRECT_FADE_NEAR_WHITE_THRESHOLD
            or raw.brightness
            >= max(ContentFilterThresholds.DIRECT_FADE_BRIGHTNESS_THRESHOLD, profile.brightness.p90)
        )
        dark_fade = (
            raw.near_black_ratio >= ContentFilterThresholds.DIRECT_FADE_NEAR_BLACK_THRESHOLD
            or raw.brightness
            <= min(ContentFilterThresholds.DIRECT_FADE_DARKNESS_THRESHOLD, profile.brightness.p10)
        )
        weak_structure = (
            raw.dominant_tone_ratio >= ContentFilterThresholds.DIRECT_FADE_MIN_DOMINANT_TONE_RATIO
            or raw.luminance_range
            <= max(ContentFilterThresholds.DIRECT_FADE_MIN_LUMINANCE_RANGE, profile.luminance_range.p25)
        )
        return (
            (bright_fade or dark_fade)
            and raw.contrast <= max(ContentFilterThresholds.DIRECT_FADE_MAX_CONTRAST, profile.contrast.p25)
            and raw.edge_density
            <= max(ContentFilterThresholds.DIRECT_FADE_MAX_EDGE_DENSITY, profile.edge_density.p25)
            and weak_structure
        )

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
            if prev_next_similarity < ContentFilterThresholds.TEMPORAL_SIMILARITY_THRESHOLD:
                continue

            current_visibility = adaptive_scores[id(images[index])].visibility_score
            prev_visibility = adaptive_scores[id(images[index - 1])].visibility_score
            next_visibility = adaptive_scores[id(images[index + 1])].visibility_score
            if current_visibility + ContentFilterThresholds.TEMPORAL_VISIBILITY_MARGIN < min(
                prev_visibility, next_visibility
            ):
                rejected_indices.add(index)

        return rejected_indices
