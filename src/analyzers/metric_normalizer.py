"""Metric normalizer for image quality metrics."""

import math

from ..models.normalized_metrics import NormalizedMetrics
from ..models.raw_metrics import RawMetrics


class MetricNormalizer:
    """メトリクス正規化クラス.

    生メトリクスを0.0-1.0の範囲に正規化するシグモイド関数を提供する。
    """

    @staticmethod
    def sigmoid(x: float, center: float, steepness: float = 0.1) -> float:
        """シグモイド関数による正規化."""
        try:
            return 1 / (1 + math.exp(-steepness * (x - center)))
        except (OverflowError, ValueError):
            # exp() のオーバーフロー時は、入力値に基づいて境界値を返す
            return 1.0 if x > center else 0.0

    @classmethod
    def normalize_all(cls, raw: RawMetrics) -> NormalizedMetrics:
        """すべてのメトリクスを正規化.

        Args:
            raw: 生メトリクス

        Returns:
            NormalizedMetricsインスタンス
        """
        return NormalizedMetrics(
            blur_score=cls.sigmoid(raw.blur_score, center=500, steepness=0.005),
            contrast=cls.sigmoid(raw.contrast, center=50, steepness=0.1),
            color_richness=cls.sigmoid(raw.color_richness, center=40, steepness=0.1),
            edge_density=min(1.0, raw.edge_density * 5.0),
            dramatic_score=min(1.0, raw.dramatic_score / 100.0),
            visual_balance=raw.visual_balance / 100.0,
            action_intensity=cls.sigmoid(
                raw.action_intensity, center=30, steepness=0.2
            ),
            ui_density=cls.sigmoid(raw.ui_density, center=10, steepness=0.3),
        )
