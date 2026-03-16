from typing import Protocol

from ..analyzers.metric_calculator import MetricCalculator
from ..models.analyzed_image import AnalyzedImage


class AnalyzerLike(Protocol):
    """GameScreenPickerが必要とする最小Analyzerインターフェース."""

    @property
    def metric_calculator(self) -> MetricCalculator:
        """MetricCalculatorを返す."""

    def analyze_batch(
        self,
        paths: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> list[AnalyzedImage | None]:
        """画像をバッチ解析する."""
