"""遷移フレーム判定に使う共通メトリクス."""

from __future__ import annotations

from ..constants.transition_thresholds import TransitionThresholds
from ..models.adaptive_scores import AdaptiveScores
from ..models.layout_heuristics import LayoutHeuristics
from ..models.normalized_metrics import NormalizedMetrics
from ..models.raw_metrics import RawMetrics
from ..models.whole_input_profile import WholeInputProfile


def clamp01(value: float) -> float:
    """値を 0..1 に丸める."""
    return max(0.0, min(1.0, value))


def calculate_bright_washout_score(raw_metrics: "RawMetrics") -> float:
    """明転・白飛び寄りの washed-out 度合いを返す."""
    t = TransitionThresholds
    brightness_score = clamp01(
        (raw_metrics.brightness - t.BRIGHT_WASHOUT_BRIGHTNESS_BASE)
        / t.BRIGHT_WASHOUT_BRIGHTNESS_SCALE
    )
    contrast_penalty = 1.0 - clamp01(
        raw_metrics.contrast / t.BRIGHT_WASHOUT_CONTRAST_SCALE
    )
    edge_penalty = 1.0 - clamp01(
        raw_metrics.edge_density / t.BRIGHT_WASHOUT_EDGE_SCALE
    )
    range_penalty = 1.0 - clamp01(
        raw_metrics.luminance_range / t.BRIGHT_WASHOUT_RANGE_SCALE
    )
    return clamp01(
        t.BRIGHT_WASHOUT_WEIGHT_BRIGHTNESS * brightness_score
        + t.BRIGHT_WASHOUT_WEIGHT_NEAR_WHITE * raw_metrics.near_white_ratio
        + t.BRIGHT_WASHOUT_WEIGHT_CONTRAST * contrast_penalty
        + t.BRIGHT_WASHOUT_WEIGHT_EDGE * edge_penalty
        + t.BRIGHT_WASHOUT_WEIGHT_DOMINANT * raw_metrics.dominant_tone_ratio
        + t.BRIGHT_WASHOUT_WEIGHT_RANGE * range_penalty
    )


def calculate_system_ui_signal(heuristics: "LayoutHeuristics") -> float:
    """システムUIらしさの強いレイアウト信号を返す."""
    return max(
        heuristics.menu_layout_score,
        heuristics.title_layout_score,
        heuristics.game_over_layout_score,
    )


def calculate_support_ui_score(normalized_metrics: "NormalizedMetrics") -> float:
    """support UI 寄りの見た目を返す."""
    t = TransitionThresholds
    return clamp01(
        t.SUPPORT_UI_DENSITY_WEIGHT * normalized_metrics.ui_density
        + t.SUPPORT_UI_ACTION_INVERSE_WEIGHT
        * (1.0 - normalized_metrics.action_intensity)
    )


def calculate_veiled_transition_score(
    raw_metrics: "RawMetrics",
    adaptive_scores: "AdaptiveScores",
    heuristics: "LayoutHeuristics",
    normalized_metrics: "NormalizedMetrics",
) -> float:
    """明転・暗転・veiled UI を含む遷移途中らしさを返す."""
    t = TransitionThresholds
    exposure_extreme = max(raw_metrics.near_black_ratio, raw_metrics.near_white_ratio)
    range_penalty = clamp01(
        1.0 - raw_metrics.luminance_range / t.BRIGHT_WASHOUT_RANGE_SCALE
    )
    contrast_penalty = clamp01(
        1.0 - raw_metrics.contrast / t.BRIGHT_WASHOUT_CONTRAST_SCALE
    )
    edge_penalty = clamp01(
        1.0 - raw_metrics.edge_density / t.BRIGHT_WASHOUT_EDGE_SCALE
    )
    bright_washout_score = calculate_bright_washout_score(raw_metrics)
    system_ui_signal = calculate_system_ui_signal(heuristics)
    support_ui_score = calculate_support_ui_score(normalized_metrics)
    return clamp01(
        t.VEILED_VISIBILITY_INVERSE_WEIGHT * (1.0 - adaptive_scores.visibility_score)
        + t.VEILED_INFORMATION_INVERSE_WEIGHT
        * (1.0 - adaptive_scores.information_score)
        + t.VEILED_EXPOSURE_EXTREME_WEIGHT * exposure_extreme
        + t.VEILED_RANGE_PENALTY_WEIGHT * range_penalty
        + t.VEILED_CONTRAST_PENALTY_WEIGHT * contrast_penalty
        + t.VEILED_EDGE_PENALTY_WEIGHT * edge_penalty
        + t.VEILED_BRIGHT_WASHOUT_WEIGHT * bright_washout_score
        + t.VEILED_SYSTEM_UI_WEIGHT * system_ui_signal
        + t.VEILED_SUPPORT_UI_WEIGHT * support_ui_score
    )


def calculate_relative_transition_scores(
    raw_metrics: "RawMetrics",
    whole_input_profile: "WholeInputProfile",
) -> tuple[float, float, float, str]:
    """入力全体分布に対する明転・暗転 outlier スコアを返す."""
    t = TransitionThresholds
    bright_tail_width = max(
        t.RELATIVE_TAIL_WIDTH_MIN,
        whole_input_profile.brightness.p90 - whole_input_profile.brightness.p50,
    )
    dark_tail_width = max(
        t.RELATIVE_TAIL_WIDTH_MIN,
        whole_input_profile.brightness.p50 - whole_input_profile.brightness.p10,
    )
    bright_outlier = clamp01(
        (raw_metrics.brightness - whole_input_profile.brightness.p90)
        / bright_tail_width
    )
    dark_outlier = clamp01(
        (whole_input_profile.brightness.p10 - raw_metrics.brightness)
        / dark_tail_width
    )
    near_white_outlier = clamp01(
        (
            raw_metrics.near_white_ratio
            - max(
                t.RELATIVE_NEAR_WHITE_FLOOR,
                whole_input_profile.near_white_ratio.p90,
            )
        )
        / t.RELATIVE_BRIGHT_OUTLIER_SCALE
    )
    near_black_outlier = clamp01(
        (
            raw_metrics.near_black_ratio
            - max(
                t.RELATIVE_NEAR_BLACK_FLOOR,
                whole_input_profile.near_black_ratio.p90,
            )
        )
        / t.RELATIVE_DARK_OUTLIER_SCALE
    )
    dominant_tone_outlier = clamp01(
        (
            raw_metrics.dominant_tone_ratio
            - max(
                t.RELATIVE_DOMINANT_TONE_FLOOR,
                whole_input_profile.dominant_tone_ratio.p90,
            )
        )
        / t.RELATIVE_DOMINANT_TONE_SCALE
    )
    contrast_penalty = clamp01(
        1.0 - raw_metrics.contrast / t.BRIGHT_WASHOUT_CONTRAST_SCALE
    )
    edge_penalty = clamp01(
        1.0 - raw_metrics.edge_density / t.BRIGHT_WASHOUT_EDGE_SCALE
    )
    range_penalty = clamp01(
        1.0 - raw_metrics.luminance_range / t.BRIGHT_WASHOUT_RANGE_SCALE
    )
    structure_loss = clamp01(
        t.STRUCTURE_CONTRAST_WEIGHT * contrast_penalty
        + t.STRUCTURE_EDGE_WEIGHT * edge_penalty
        + t.STRUCTURE_RANGE_WEIGHT * range_penalty
    )
    bright_washout_score = calculate_bright_washout_score(raw_metrics)
    relative_bright_transition_score = clamp01(
        t.RELATIVE_BRIGHT_OUTLIER_WEIGHT * bright_outlier
        + t.RELATIVE_BRIGHT_NEAR_WHITE_WEIGHT * near_white_outlier
        + t.RELATIVE_BRIGHT_WASHOUT_WEIGHT * bright_washout_score
        + t.RELATIVE_BRIGHT_STRUCTURE_WEIGHT * structure_loss
    )
    relative_dark_transition_score = clamp01(
        t.RELATIVE_DARK_OUTLIER_WEIGHT * dark_outlier
        + t.RELATIVE_DARK_NEAR_BLACK_WEIGHT * near_black_outlier
        + t.RELATIVE_DARK_DOMINANT_WEIGHT * dominant_tone_outlier
        + t.RELATIVE_DARK_STRUCTURE_WEIGHT * structure_loss
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
