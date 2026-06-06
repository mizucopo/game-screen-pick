"""呼び出し回数を記録するテスト用Analyzer."""

from collections.abc import Callable
from types import SimpleNamespace

from src.analyzers.metric_calculator import MetricCalculator
from src.models.analyzed_image import AnalyzedImage
from src.models.analyzer_config import AnalyzerConfig
from tests.conftest import _feature, create_analyzed_image


class FakeCountingAnalyzer:
    """解析呼び出しを記録する最小フェイク."""

    def __init__(
        self,
        model_name: str = "clip-a",
        config: AnalyzerConfig | None = None,
    ) -> None:
        self.metric_calculator = MetricCalculator(AnalyzerConfig())
        self.config = config or AnalyzerConfig()
        self.feature_extractor = SimpleNamespace(
            model_manager=SimpleNamespace(model_name=model_name),
        )
        self.requested_paths: list[list[str]] = []

    def analyze_batch(
        self,
        paths: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
        on_chunk_processed: Callable[[list[AnalyzedImage | None]], None] | None = None,
    ) -> list[AnalyzedImage | None]:
        """中立解析結果を返し、呼び出しpathを記録する."""
        del batch_size, show_progress
        self.requested_paths.append(paths)
        results: list[AnalyzedImage | None] = [
            create_analyzed_image(path=path, combined_features=_feature(index))
            for index, path in enumerate(paths)
        ]
        if on_chunk_processed is not None:
            on_chunk_processed(results)
        return results
