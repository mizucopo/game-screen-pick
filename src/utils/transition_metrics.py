"""遷移フレーム判定に使う共通メトリクス."""

from __future__ import annotations

from ..constants.transition_thresholds import (
    BRIGHT_WASHOUT_BRIGHTNESS_BASE,
    BRIGHT_WASHOUT_BRIGHTNESS_SCALE,
    BRIGHT_WASHOUT_CONTRAST_SCALE,
    BRIGHT_WASHOUT_EDGE_SCALE,
    BRIGHT_WASHOUT_RANGE_SCALE,
    BRIGHT_WASHOUT_WEIGHT_BRIGHTNESS,
    BRIGHT_WASHOUT_WEIGHT_CONTRAST,
    BRIGHT_WASHOUT_WEIGHT_DOMINANT,
    BRIGHT_WASHOUT_WEIGHT_EDGE,
    BRIGHT_WASHOUT_WEIGHT_NEAR_WHITE,
    BRIGHT_WASHOUT_WEIGHT_RANGE,
    RELATIVE_BRIGHT_NEAR_WHITE_WEIGHT,
    RELATIVE_BRIGHT_OUTLIER_SCALE,
    RELATIVE_BRIGHT_OUTLIER_WEIGHT,
    RELATIVE_BRIGHT_STRUCTURE_WEIGHT,
    RELATIVE_BRIGHT_WASHOUT_WEIGHT,
    RELATIVE_DARK_DOMINANT_WEIGHT,
    RELATIVE_DARK_NEAR_BLACK_WEIGHT,
    RELATIVE_DARK_OUTLIER_SCALE,
    RELATIVE_DARK_OUTLIER_WEIGHT,
    RELATIVE_DARK_STRUCTURE_WEIGHT,
    RELATIVE_DOMINANT_TONE_FLOOR,
    RELATIVE_DOMINANT_TONE_SCALE,
    RELATIVE_NEAR_BLACK_FLOOR,
    RELATIVE_NEAR_WHITE_FLOOR,
    RELATIVE_TAIL_WIDTH_MIN,
    STRUCTURE_CONTRAST_WEIGHT,
    STRUCTURE_EDGE_WEIGHT,
    STRUCTURE_RANGE_WEIGHT,
    SUPPORT_UI_ACTION_INVERSE_WEIGHT,
    SUPPORT_UI_DENSITY_WEIGHT,
    VEILED_BRIGHT_WASHOUT_WEIGHT,
    VEILED_CONTRAST_PENALTY_WEIGHT,
    VEILED_EDGE_PENALTY_WEIGHT,
    VEILED_EXPOSURE_EXTREME_WEIGHT,
    VEILED_INFORMATION_INVERSE_WEIGHT,
    VEILED_RANGE_PENALTY_WEIGHT,
    VEILED_SUPPORT_UI_WEIGHT,
    VEILED_SYSTEM_UI_WEIGHT,
    VEILED_VISIBILITY_INVERSE_WEIGHT,
)
from ..models.adaptive_scores import AdaptiveScores
from ..models.layout_heuristics import LayoutHeuristics
from ..models.normalized_metrics import NormalizedMetrics
from ..models.raw_metrics import RawMetrics
from ..models.whole_input_profile import WholeInputProfile


class TransitionMetrics:
    """遷移メトリクス計算ユーティリティクラス."""

    @staticmethod
    def _clamp01(value: float) -> float:
        """値を 0..1 に丸める."""
        return max(0.0, min(1.0, value))

    @staticmethod
    def _calculate_support_ui_score(normalized_metrics: NormalizedMetrics) -> float:
        """support UI 寄りの見た目を返す."""
        return TransitionMetrics._clamp01(
            SUPPORT_UI_DENSITY_WEIGHT * normalized_metrics.ui_density
            + SUPPORT_UI_ACTION_INVERSE_WEIGHT
            * (1.0 - normalized_metrics.action_intensity)
        )

    @staticmethod
    def calculate_bright_washout_score(raw_metrics: RawMetrics) -> float:
        """明転・白飛び寄りの washed-out 度合いを返す."""
        brightness_score = TransitionMetrics._clamp01(
            (raw_metrics.brightness - BRIGHT_WASHOUT_BRIGHTNESS_BASE)
            / BRIGHT_WASHOUT_BRIGHTNESS_SCALE
        )
        contrast_penalty = 1.0 - TransitionMetrics._clamp01(
            raw_metrics.contrast / BRIGHT_WASHOUT_CONTRAST_SCALE
        )
        edge_penalty = 1.0 - TransitionMetrics._clamp01(
            raw_metrics.edge_density / BRIGHT_WASHOUT_EDGE_SCALE
        )
        range_penalty = TransitionMetrics._clamp01(
            1.0 - raw_metrics.luminance_range / BRIGHT_WASHOUT_RANGE_SCALE
        )
        return TransitionMetrics._clamp01(
            BRIGHT_WASHOUT_WEIGHT_BRIGHTNESS * brightness_score
            + BRIGHT_WASHOUT_WEIGHT_NEAR_WHITE * raw_metrics.near_white_ratio
            + BRIGHT_WASHOUT_WEIGHT_CONTRAST * contrast_penalty
            + BRIGHT_WASHOUT_WEIGHT_EDGE * edge_penalty
            + BRIGHT_WASHOUT_WEIGHT_DOMINANT * raw_metrics.dominant_tone_ratio
            + BRIGHT_WASHOUT_WEIGHT_RANGE * range_penalty
        )

    @staticmethod
    def calculate_system_ui_signal(heuristics: LayoutHeuristics) -> float:
        """システムUIらしさの強いレイアウト信号を返す."""
        return max(
            heuristics.menu_layout_score,
            heuristics.title_layout_score,
            heuristics.game_over_layout_score,
        )

    @staticmethod
    def calculate_veiled_transition_score(
        raw_metrics: RawMetrics,
        adaptive_scores: AdaptiveScores,
        heuristics: LayoutHeuristics,
        normalized_metrics: NormalizedMetrics,
        bright_washout_score: float | None = None,
    ) -> float:
        """明転・暗転・veiled UI を含む遷移途中らしさを返す."""
        exposure_extreme = max(
            raw_metrics.near_black_ratio, raw_metrics.near_white_ratio
        )
        range_penalty = TransitionMetrics._clamp01(
            1.0 - raw_metrics.luminance_range / BRIGHT_WASHOUT_RANGE_SCALE
        )
        contrast_penalty = TransitionMetrics._clamp01(
            1.0 - raw_metrics.contrast / BRIGHT_WASHOUT_CONTRAST_SCALE
        )
        edge_penalty = TransitionMetrics._clamp01(
            1.0 - raw_metrics.edge_density / BRIGHT_WASHOUT_EDGE_SCALE
        )
        _bright_washout_score = (
            bright_washout_score
            if bright_washout_score is not None
            else TransitionMetrics.calculate_bright_washout_score(raw_metrics)
        )
        system_ui_signal = TransitionMetrics.calculate_system_ui_signal(heuristics)
        support_ui_score = TransitionMetrics._calculate_support_ui_score(
            normalized_metrics
        )
        return TransitionMetrics._clamp01(
            VEILED_VISIBILITY_INVERSE_WEIGHT * (1.0 - adaptive_scores.visibility_score)
            + VEILED_INFORMATION_INVERSE_WEIGHT
            * (1.0 - adaptive_scores.information_score)
            + VEILED_EXPOSURE_EXTREME_WEIGHT * exposure_extreme
            + VEILED_RANGE_PENALTY_WEIGHT * range_penalty
            + VEILED_CONTRAST_PENALTY_WEIGHT * contrast_penalty
            + VEILED_EDGE_PENALTY_WEIGHT * edge_penalty
            + VEILED_BRIGHT_WASHOUT_WEIGHT * _bright_washout_score
            + VEILED_SYSTEM_UI_WEIGHT * system_ui_signal
            + VEILED_SUPPORT_UI_WEIGHT * support_ui_score
        )

    @staticmethod
    def calculate_relative_transition_scores(
        raw_metrics: RawMetrics,
        whole_input_profile: WholeInputProfile,
    ) -> tuple[float, float, float, str]:
        """入力全体分布に対する明転・暗転 outlier スコアを返す."""
        bright_tail_width = max(
            RELATIVE_TAIL_WIDTH_MIN,
            whole_input_profile.brightness.p90 - whole_input_profile.brightness.p50,
        )
        dark_tail_width = max(
            RELATIVE_TAIL_WIDTH_MIN,
            whole_input_profile.brightness.p50 - whole_input_profile.brightness.p10,
        )
        bright_outlier = TransitionMetrics._clamp01(
            (raw_metrics.brightness - whole_input_profile.brightness.p90)
            / bright_tail_width
        )
        dark_outlier = TransitionMetrics._clamp01(
            (whole_input_profile.brightness.p10 - raw_metrics.brightness)
            / dark_tail_width
        )
        near_white_outlier = TransitionMetrics._clamp01(
            (
                raw_metrics.near_white_ratio
                - max(
                    RELATIVE_NEAR_WHITE_FLOOR,
                    whole_input_profile.near_white_ratio.p90,
                )
            )
            / RELATIVE_BRIGHT_OUTLIER_SCALE
        )
        near_black_outlier = TransitionMetrics._clamp01(
            (
                raw_metrics.near_black_ratio
                - max(
                    RELATIVE_NEAR_BLACK_FLOOR,
                    whole_input_profile.near_black_ratio.p90,
                )
            )
            / RELATIVE_DARK_OUTLIER_SCALE
        )
        dominant_tone_outlier = TransitionMetrics._clamp01(
            (
                raw_metrics.dominant_tone_ratio
                - max(
                    RELATIVE_DOMINANT_TONE_FLOOR,
                    whole_input_profile.dominant_tone_ratio.p90,
                )
            )
            / RELATIVE_DOMINANT_TONE_SCALE
        )
        contrast_penalty = TransitionMetrics._clamp01(
            1.0 - raw_metrics.contrast / BRIGHT_WASHOUT_CONTRAST_SCALE
        )
        edge_penalty = TransitionMetrics._clamp01(
            1.0 - raw_metrics.edge_density / BRIGHT_WASHOUT_EDGE_SCALE
        )
        range_penalty = TransitionMetrics._clamp01(
            1.0 - raw_metrics.luminance_range / BRIGHT_WASHOUT_RANGE_SCALE
        )
        structure_loss = TransitionMetrics._clamp01(
            STRUCTURE_CONTRAST_WEIGHT * contrast_penalty
            + STRUCTURE_EDGE_WEIGHT * edge_penalty
            + STRUCTURE_RANGE_WEIGHT * range_penalty
        )
        bright_washout_score = TransitionMetrics.calculate_bright_washout_score(
            raw_metrics
        )
        relative_bright_transition_score = TransitionMetrics._clamp01(
            RELATIVE_BRIGHT_OUTLIER_WEIGHT * bright_outlier
            + RELATIVE_BRIGHT_NEAR_WHITE_WEIGHT * near_white_outlier
            + RELATIVE_BRIGHT_WASHOUT_WEIGHT * bright_washout_score
            + RELATIVE_BRIGHT_STRUCTURE_WEIGHT * structure_loss
        )
        relative_dark_transition_score = TransitionMetrics._clamp01(
            RELATIVE_DARK_OUTLIER_WEIGHT * dark_outlier
            + RELATIVE_DARK_NEAR_BLACK_WEIGHT * near_black_outlier
            + RELATIVE_DARK_DOMINANT_WEIGHT * dominant_tone_outlier
            + RELATIVE_DARK_STRUCTURE_WEIGHT * structure_loss
        )
        relative_transition_score = max(
            relative_bright_transition_score, relative_dark_transition_score
        )
        relative_transition_polarity = (
            "bright"
            if relative_bright_transition_score >= relative_dark_transition_score
            else "dark"
        )
        return (
            relative_bright_transition_score,
            relative_dark_transition_score,
            relative_transition_score,
            relative_transition_polarity,
        )
