"""中立画像解析結果の再開cache."""

import hashlib
import pickle
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from ..models.analyzed_image import AnalyzedImage


class NeutralAnalysisCache:
    """中立画像解析結果を入力フォルダ配下へ保存する."""

    VERSION = "neutral-analysis-v1"

    def __init__(self, input_folder: Path, analyzer_fingerprint: str) -> None:
        """cacheを初期化する."""
        self._cache_dir = (
            input_folder / ".game-screen-pick" / "cache" / "neutral-analysis"
        )
        self._analyzer_fingerprint = analyzer_fingerprint

    def read(self, image_path: Path) -> AnalyzedImage | None:
        """画像pathに対応するcache済み解析結果を返す."""
        cache_path = self._cache_path(image_path)
        if not cache_path.exists():
            return None
        try:
            with cache_path.open("rb") as cache_file:
                cached = pickle.load(cache_file)
        except (OSError, pickle.PickleError, EOFError, AttributeError, ValueError):
            return None
        if not isinstance(cached, AnalyzedImage):
            return None
        if cached.path != str(image_path):
            return None
        return cached

    def write_many(self, analyzed_images: list[AnalyzedImage]) -> None:
        """複数の解析結果をcacheへ保存する."""
        for analyzed_image in analyzed_images:
            self._write(analyzed_image)

    def _write(self, analyzed_image: AnalyzedImage) -> None:
        """単一の解析結果をatomicにcacheへ保存する."""
        image_path = Path(analyzed_image.path)
        try:
            cache_path = self._cache_path(image_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(
                "wb",
                dir=cache_path.parent,
                delete=False,
            ) as cache_file:
                pickle.dump(analyzed_image, cache_file)
                temp_path = Path(cache_file.name)
            temp_path.replace(cache_path)
        except OSError:
            return

    def _cache_path(self, image_path: Path) -> Path:
        """画像pathからcache file pathを返す."""
        return self._cache_dir / f"{self._cache_key(image_path)}.pickle"

    def _cache_key(self, image_path: Path) -> str:
        """画像pathと解析設定からcache keyを作る."""
        stat = image_path.stat()
        payload: dict[str, Any] = {
            "version": self.VERSION,
            "path": str(image_path.resolve()),
            "mtime_ns": stat.st_mtime_ns,
            "size": stat.st_size,
            "analyzer": self._analyzer_fingerprint,
        }
        raw_key = repr(sorted(payload.items()))
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
