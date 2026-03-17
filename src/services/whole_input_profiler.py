"""入力全体の分布と相対スコアを計算する."""

from typing import Any

import numpy as np

from ..models.adaptive_scores import AdaptiveScores
from ..models.analyzed_image import AnalyzedImage
from ..models.metric_distribution import MetricDistribution
from ..models.whole_input_profile import WholeInputProfile


class WholeInputProfiler:
    """入力全体の分布プロフィールと相対スコアを算出する."""

    INFORMATION_WEIGHTS = {
        "contrast": 0.20,
        "edge_density": 0.25,
        "action_intensity": 0.15,
        "luminance_entropy": 0.20,
        "luminance_range": 0.20,
    }
    VISIBILITY_RANGE_SCALE = 32.0
    VISIBILITY_ENTROPY_SCALE = 1.5
    VISIBILITY_EDGE_SCALE = 0.08
    VISIBILITY_CONTRAST_SCALE = 12.0

    def build_profile(self, images: list[AnalyzedImage]) -> WholeInputProfile:
        """入力全体から主要メトリクスの分布を作る."""
        if not images:
            empty = MetricDistribution(0.0, 0.0, 0.0, 0.0)
            return WholeInputProfile(
                brightness=empty,
                contrast=empty,
                edge_density=empty,
                action_intensity=empty,
                luminance_entropy=empty,
                luminance_range=empty,
                near_black_ratio=empty,
                near_white_ratio=empty,
                dominant_tone_ratio=empty,
            )

        return WholeInputProfile(
            brightness=self._build_distribution(
                [image.raw_metrics.brightness for image in images]
            ),
            contrast=self._build_distribution(
                [image.raw_metrics.contrast for image in images]
            ),
            edge_density=self._build_distribution(
                [image.raw_metrics.edge_density for image in images]
            ),
            action_intensity=self._build_distribution(
                [image.raw_metrics.action_intensity for image in images]
            ),
            luminance_entropy=self._build_distribution(
                [image.raw_metrics.luminance_entropy for image in images]
            ),
            luminance_range=self._build_distribution(
                [image.raw_metrics.luminance_range for image in images]
            ),
            near_black_ratio=self._build_distribution(
                [image.raw_metrics.near_black_ratio for image in images]
            ),
            near_white_ratio=self._build_distribution(
                [image.raw_metrics.near_white_ratio for image in images]
            ),
            dominant_tone_ratio=self._build_distribution(
                [image.raw_metrics.dominant_tone_ratio for image in images]
            ),
        )

    def score_images(
        self,
        images: list[AnalyzedImage],
    ) -> dict[int, AdaptiveScores]:
        """入力全体を見た相対情報量と差分量を返す."""
        if not images:
            return {}

        information_scores = self._calculate_information_scores(images)
        visibility_scores = self._calculate_visibility_scores(images)
        return {
            id(image): AdaptiveScores(
                information_score=information_scores[id(image)],
                visibility_score=visibility_scores[id(image)],
            )
            for image in images
        }

    @staticmethod
    def _build_distribution(values: list[float]) -> MetricDistribution:
        """主要パーセンタイルだけをまとめる."""
        if not values:
            return MetricDistribution(0.0, 0.0, 0.0, 0.0)

        array = np.asarray(values, dtype=np.float32)
        p10, p25, p50, p90 = np.percentile(array, [10, 25, 50, 90])
        return MetricDistribution(
            p10=float(p10),
            p25=float(p25),
            p50=float(p50),
            p90=float(p90),
        )

    def _calculate_information_scores(
        self,
        images: list[AnalyzedImage],
    ) -> dict[int, float]:
        """入力分布に対する相対的な情報量スコアを返す."""
        metric_values = {
            metric_name: np.sort(
                np.asarray(
                    [getattr(image.raw_metrics, metric_name) for image in images],
                    dtype=np.float32,
                )
            )
            for metric_name in self.INFORMATION_WEIGHTS
        }

        information_scores: dict[int, float] = {}
        for image in images:
            score = 0.0
            for metric_name, weight in self.INFORMATION_WEIGHTS.items():
                value = getattr(image.raw_metrics, metric_name)
                score += weight * self._percentile_rank(
                    value, metric_values[metric_name]
                )
            information_scores[id(image)] = float(score)
        return information_scores

    def _calculate_visibility_scores(
        self,
        images: list[AnalyzedImage],
    ) -> dict[int, float]:
        """絶対的な見えやすさスコアを返す."""
        visibility_scores: dict[int, float] = {}
        for image in images:
            raw = image.raw_metrics
            visibility_scores[id(image)] = float(
                0.30 * min(1.0, raw.luminance_range / self.VISIBILITY_RANGE_SCALE)
                + 0.25 * min(1.0, raw.luminance_entropy / self.VISIBILITY_ENTROPY_SCALE)
                + 0.20 * min(1.0, raw.edge_density / self.VISIBILITY_EDGE_SCALE)
                + 0.15 * min(1.0, raw.contrast / self.VISIBILITY_CONTRAST_SCALE)
                + 0.10 * (1.0 - max(raw.near_black_ratio, raw.near_white_ratio))
            )
        return visibility_scores

    @staticmethod
    def _percentile_rank(value: float, sorted_values: np.ndarray[Any, Any]) -> float:
        """同値を平均順位で扱う percentile rank を返す."""
        if sorted_values.size == 0:
            return 0.0

        left_index = np.searchsorted(sorted_values, value, side="left")
        right_index = np.searchsorted(sorted_values, value, side="right")
        average_rank = (left_index + right_index) / 2.0
        return float(average_rank / sorted_values.size)
