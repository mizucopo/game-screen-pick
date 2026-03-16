"""入力全体分布の要約."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricDistribution:
    """単一メトリクスの主要パーセンタイル."""

    p10: float
    p25: float
    p50: float
    p90: float
