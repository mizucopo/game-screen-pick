"""解析中に入力画像を書き換えるテスト用Analyzer."""

from collections.abc import Callable
from pathlib import Path

from src.models.analyzed_image import AnalyzedImage
from tests.conftest import _feature, create_analyzed_image
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
        del batch_size, show_progress
        self.requested_paths.append(paths)
        results: list[AnalyzedImage | None] = [
            create_analyzed_image(path=path, combined_features=_feature(index))
            for index, path in enumerate(paths)
        ]
        for path in paths:
            Path(path).write_bytes(b"replaced-image")
        if on_chunk_processed is not None:
            on_chunk_processed(results)
        return results
