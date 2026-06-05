"""中立画像解析結果の再開cache."""

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from ..models.analyzed_image import AnalyzedImage
from ..models.layout_heuristics import LayoutHeuristics
from ..models.normalized_metrics import NormalizedMetrics
from ..models.raw_metrics import RawMetrics

type _FileVersion = tuple[str, int, int]


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
                restored, stored_resolved_path = self._restore_analyzed_image(cached)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
        if stored_resolved_path != self._resolved_path(image_path):
            return None
        return replace(restored, path=str(image_path))

    def write_many(
        self,
        analyzed_images: list[AnalyzedImage],
        expected_versions: dict[str, _FileVersion] | None = None,
    ) -> None:
        """複数の解析結果をcacheへ保存する."""
        for analyzed_image in analyzed_images:
            expected_version = (
                expected_versions.get(analyzed_image.path)
                if expected_versions is not None
                else None
            )
            self._write(analyzed_image, expected_version)

    def capture_versions(self, image_paths: list[Path]) -> dict[str, _FileVersion]:
        """画像pathごとの現在file versionを返す."""
        versions: dict[str, _FileVersion] = {}
        for image_path in image_paths:
            try:
                versions[str(image_path)] = self._file_version(image_path)
            except OSError:
                continue
        return versions

    def _write(
        self,
        analyzed_image: AnalyzedImage,
        expected_version: _FileVersion | None = None,
    ) -> None:
        """単一の解析結果をatomicにcacheへ保存する."""
        image_path = Path(analyzed_image.path)
        try:
            current_version = self._file_version(image_path)
            if expected_version is not None and current_version != expected_version:
                return
            cache_path = self._cache_path_from_version(current_version)
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
        return self._cache_path_from_version(self._file_version(image_path))

    def _cache_path_from_version(self, file_version: _FileVersion) -> Path:
        """file versionからcache file pathを返す."""
        return self._cache_dir / f"{self._cache_key_from_version(file_version)}.npz"

    def _try_cache_path(self, image_path: Path) -> Path | None:
        """cache file pathを返し、path情報を読めない場合はNoneを返す."""
        try:
            return self._cache_path(image_path)
        except OSError:
            return None

    def _cache_key_from_version(self, file_version: _FileVersion) -> str:
        """file versionと解析設定からcache keyを作る."""
        resolved_path, mtime_ns, size = file_version
        payload: dict[str, object] = {
            "version": self.VERSION,
            "path": resolved_path,
            "mtime_ns": mtime_ns,
            "size": size,
            "analyzer": self._analyzer_fingerprint,
        }
        raw_key = repr(sorted(payload.items()))
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    @staticmethod
    def _file_version(image_path: Path) -> _FileVersion:
        """cache keyに使う画像file versionを返す."""
        stat = image_path.stat()
        return (
            NeutralAnalysisCache._resolved_path(image_path),
            stat.st_mtime_ns,
            stat.st_size,
        )

    @staticmethod
    def _resolved_path(path: str | Path) -> str:
        """path表記差を吸収するためresolve済みpathを返す."""
        return str(Path(path).resolve())

    @staticmethod
    def _restore_analyzed_image(
        cached: np.lib.npyio.NpzFile,
    ) -> tuple[AnalyzedImage, str]:
        """npz cache payloadから中立解析結果を復元する."""
        payload = json.loads(str(cached["metadata"].item()))
        restored = AnalyzedImage(
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
        fallback_resolved_path = NeutralAnalysisCache._resolved_path(restored.path)
        stored_resolved_path = str(payload.get("resolved_path", fallback_resolved_path))
        return restored, stored_resolved_path

    @staticmethod
    def _metadata_payload(analyzed_image: AnalyzedImage) -> dict[str, object]:
        """配列以外の解析結果をJSON保存用payloadへ変換する."""
        return {
            "path": analyzed_image.path,
            "resolved_path": NeutralAnalysisCache._resolved_path(analyzed_image.path),
            "raw_metrics": asdict(analyzed_image.raw_metrics),
            "normalized_metrics": asdict(analyzed_image.normalized_metrics),
            "layout_heuristics": asdict(analyzed_image.layout_heuristics),
        }
