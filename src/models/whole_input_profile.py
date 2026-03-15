"""入力全体分布プロフィール."""

from dataclasses import dataclass

from .metric_distribution import MetricDistribution


@dataclass(frozen=True)
class WholeInputProfile:
    """入力全体の画面傾向を表す分布プロフィール."""

    brightness: MetricDistribution
    contrast: MetricDistribution
    edge_density: MetricDistribution
    action_intensity: MetricDistribution
    luminance_entropy: MetricDistribution
    luminance_range: MetricDistribution
