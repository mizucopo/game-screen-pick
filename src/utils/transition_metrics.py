"""遷移フレーム判定に使う共通メトリクス."""

from typing import Any


def clamp01(value: float) -> float:
    """値を 0..1 に丸める."""
    return max(0.0, min(1.0, value))


def calculate_bright_washout_score(raw_metrics: Any) -> float:
    """明転・白飛び寄りの washed-out 度合いを返す."""
    brightness_score = clamp01((raw_metrics.brightness - 180.0) / 75.0)
    contrast_penalty = 1.0 - clamp01(raw_metrics.contrast / 18.0)
    edge_penalty = 1.0 - clamp01(raw_metrics.edge_density / 0.14)
    range_penalty = 1.0 - clamp01(raw_metrics.luminance_range / 48.0)
    return clamp01(
        0.25 * brightness_score
        + 0.20 * raw_metrics.near_white_ratio
        + 0.20 * contrast_penalty
        + 0.15 * edge_penalty
        + 0.10 * raw_metrics.dominant_tone_ratio
        + 0.10 * range_penalty
    )


def calculate_system_ui_signal(heuristics: Any) -> float:
    """システムUIらしさの強いレイアウト信号を返す."""
    return max(
        heuristics.menu_layout_score,
        heuristics.title_layout_score,
        heuristics.game_over_layout_score,
    )


def calculate_support_ui_score(normalized_metrics: Any) -> float:
    """support UI 寄りの見た目を返す."""
    return clamp01(
        0.65 * normalized_metrics.ui_density
        + 0.35 * (1.0 - normalized_metrics.action_intensity)
    )


def calculate_veiled_transition_score(
    raw_metrics: Any,
    adaptive_scores: Any,
    heuristics: Any,
    normalized_metrics: Any,
) -> float:
    """明転・暗転・veiled UI を含む遷移途中らしさを返す."""
    exposure_extreme = max(raw_metrics.near_black_ratio, raw_metrics.near_white_ratio)
    range_penalty = clamp01(1.0 - raw_metrics.luminance_range / 48.0)
    contrast_penalty = clamp01(1.0 - raw_metrics.contrast / 18.0)
    edge_penalty = clamp01(1.0 - raw_metrics.edge_density / 0.14)
    bright_washout_score = calculate_bright_washout_score(raw_metrics)
    system_ui_signal = calculate_system_ui_signal(heuristics)
    support_ui_score = calculate_support_ui_score(normalized_metrics)
    return clamp01(
        0.18 * (1.0 - adaptive_scores.visibility_score)
        + 0.16 * (1.0 - adaptive_scores.information_score)
        + 0.12 * exposure_extreme
        + 0.10 * range_penalty
        + 0.10 * contrast_penalty
        + 0.08 * edge_penalty
        + 0.10 * bright_washout_score
        + 0.08 * system_ui_signal
        + 0.08 * support_ui_score
    )


def calculate_relative_transition_scores(
    raw_metrics: Any,
    adaptive_scores: Any,
    heuristics: Any,
    normalized_metrics: Any,
    whole_input_profile: Any,
) -> tuple[float, float, float, str]:
    """入力全体分布に対する明転・暗転 outlier スコアを返す."""
    bright_tail_width = max(
        8.0,
        whole_input_profile.brightness.p90 - whole_input_profile.brightness.p50,
    )
    dark_tail_width = max(
        8.0,
        whole_input_profile.brightness.p50 - whole_input_profile.brightness.p10,
    )
    bright_outlier = clamp01(
        (raw_metrics.brightness - whole_input_profile.brightness.p90)
        / bright_tail_width
    )
    dark_outlier = clamp01(
        (whole_input_profile.brightness.p10 - raw_metrics.brightness) / dark_tail_width
    )
    near_white_outlier = clamp01(
        (
            raw_metrics.near_white_ratio
            - max(0.05, whole_input_profile.near_white_ratio.p90)
        )
        / 0.18
    )
    near_black_outlier = clamp01(
        (
            raw_metrics.near_black_ratio
            - max(0.05, whole_input_profile.near_black_ratio.p90)
        )
        / 0.18
    )
    dominant_tone_outlier = clamp01(
        (
            raw_metrics.dominant_tone_ratio
            - max(0.55, whole_input_profile.dominant_tone_ratio.p90)
        )
        / 0.20
    )
    contrast_penalty = clamp01(1.0 - raw_metrics.contrast / 18.0)
    edge_penalty = clamp01(1.0 - raw_metrics.edge_density / 0.14)
    range_penalty = clamp01(1.0 - raw_metrics.luminance_range / 48.0)
    structure_loss = clamp01(
        0.40 * contrast_penalty + 0.35 * edge_penalty + 0.25 * range_penalty
    )
    bright_washout_score = calculate_bright_washout_score(raw_metrics)
    relative_bright_transition_score = clamp01(
        0.35 * bright_outlier
        + 0.25 * near_white_outlier
        + 0.20 * bright_washout_score
        + 0.20 * structure_loss
    )
    relative_dark_transition_score = clamp01(
        0.40 * dark_outlier
        + 0.25 * near_black_outlier
        + 0.20 * dominant_tone_outlier
        + 0.15 * structure_loss
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
