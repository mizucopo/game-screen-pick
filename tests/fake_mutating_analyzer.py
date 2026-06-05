"""解析中に入力画像を書き換えるテスト用Analyzer."""

from collections.abc import Callable
from pathlib import Path

from src.models.analyzed_image import AnalyzedImage
from tests.fake_counting_analyzer import FakeCountingAnalyzer


class FakeMutatingAnalyzer(FakeCountingAnalyzer):
    """解析完了直後に入力画像を書き換えるフェイク."""

    def analyze_batch(
        self,
        paths: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
        on_chunk_processed: Callable[[list[AnalyzedImage | None]], None] | None = None,
    ) -> list[AnalyzedImage | None]:
        """入力画像を書き換えてからchunk処理callbackを通知する."""
        results = super().analyze_batch(
            paths,
            batch_size=batch_size,
            show_progress=show_progress,
            on_chunk_processed=None,
        )
        for path in paths:
            Path(path).write_bytes(b"replaced-image")
        if on_chunk_processed is not None:
            on_chunk_processed(results)
        return results
