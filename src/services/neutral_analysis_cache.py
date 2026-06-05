"""中立画像解析結果の再開cache."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np

from ..models.analyzed_image import AnalyzedImage
from ..models.layout_heuristics import LayoutHeuristics
from ..models.normalized_metrics import NormalizedMetrics
from ..models.raw_metrics import RawMetrics


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
        cache_path = self._try_cache_path(image_path)
        if cache_path is None:
            return None
        if not cache_path.exists():
            return None
        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                restored = self._restore_analyzed_image(cached)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
        if restored.path != str(image_path):
            return None
        return restored

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
                np.savez_compressed(
                    cache_file,
                    metadata=json.dumps(
                        self._metadata_payload(analyzed_image),
                        ensure_ascii=False,
                    ),
                    clip_features=analyzed_image.clip_features,
                    combined_features=analyzed_image.combined_features,
                    content_features=analyzed_image.content_features,
                )
                temp_path = Path(cache_file.name)
            temp_path.replace(cache_path)
        except OSError:
            return

    def _cache_path(self, image_path: Path) -> Path:
        """画像pathからcache file pathを返す."""
        return self._cache_dir / f"{self._cache_key(image_path)}.npz"

    def _try_cache_path(self, image_path: Path) -> Path | None:
        """cache file pathを返し、path情報を読めない場合はNoneを返す."""
        try:
            return self._cache_path(image_path)
        except OSError:
            return None

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

    @staticmethod
    def _restore_analyzed_image(cached: np.lib.npyio.NpzFile) -> AnalyzedImage:
        """npz cache payloadから中立解析結果を復元する."""
        payload = json.loads(str(cached["metadata"].item()))
        return AnalyzedImage(
            path=str(payload["path"]),
            raw_metrics=RawMetrics(**payload["raw_metrics"]),
            normalized_metrics=NormalizedMetrics(
                **payload["normalized_metrics"],
            ),
            clip_features=cached["clip_features"],
            combined_features=cached["combined_features"],
            content_features=cached["content_features"],
            layout_heuristics=LayoutHeuristics(**payload["layout_heuristics"]),
        )

    @staticmethod
    def _metadata_payload(analyzed_image: AnalyzedImage) -> dict[str, object]:
        """配列以外の解析結果をJSON保存用payloadへ変換する."""
        return {
            "path": analyzed_image.path,
            "raw_metrics": asdict(analyzed_image.raw_metrics),
            "normalized_metrics": asdict(analyzed_image.normalized_metrics),
            "layout_heuristics": asdict(analyzed_image.layout_heuristics),
        }
