from typing import cast

from src.analyzers.metric_calculator import MetricCalculator
from src.models.analyzed_image import AnalyzedImage
from src.models.analyzer_config import AnalyzerConfig


class FakeAnalyzer:
    """GameScreenPicker向けの軽量アナライザー.

    事前に組み立てた `AnalyzedImage` を返し、
    実ファイル解析なしでピッカーのドメインロジックだけをテストする。
    """

    def __init__(self, analyzed_images: list[AnalyzedImage]) -> None:
        self._analyzed_images = analyzed_images
        self.metric_calculator = MetricCalculator(AnalyzerConfig())

    def analyze_batch(
        self,
        paths: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> list[AnalyzedImage | None]:
        del batch_size, show_progress
        return cast(list[AnalyzedImage | None], self._analyzed_images[: len(paths)])
