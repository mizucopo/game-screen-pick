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
