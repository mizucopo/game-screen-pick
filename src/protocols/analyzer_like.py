from typing import Protocol

from ..analyzers.metric_calculator import MetricCalculator
from ..models.analyzed_image import AnalyzedImage
from .text_embedding_provider import TextEmbeddingProvider


class AnalyzerLike(Protocol):
    """GameScreenPickerが必要とする最小Analyzerインターフェース.

    具体実装に依存せず、scene判定と候補採点に必要な
    最小限の公開面だけを表す。
    """

    @property
    def model_manager(self) -> TextEmbeddingProvider:
        """埋め込み取得器を返す."""

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
