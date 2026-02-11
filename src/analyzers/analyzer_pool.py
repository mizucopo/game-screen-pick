"""Multiprocessing pool for ImageQualityAnalyzer with model persistence.

各ワーカープロセスがモデルを保持することで、モデルロードコストを削減する。
"""

import logging
import os
from multiprocessing.pool import Pool
from typing import Any, List, Literal, Optional

from src.analyzers.image_quality_analyzer import ImageQualityAnalyzer
from src.models.image_metrics import ImageMetrics

logger = logging.getLogger(__name__)

# グローバル変数: 各ワーカープロセスで1回だけ初期化
_analyzer: Optional[ImageQualityAnalyzer] = None


def _init_worker(genre: str = "mixed", force_cpu: bool = False) -> None:
    """ワーカープロセスの初期化関数.

    各ワーカープロセスで1回だけ呼び出され、ImageQualityAnalyzerを初期化する。

    Args:
        genre: ジャンル重みの種類
        force_cpu: GPUを無効化してCPUを強制する（複数ワーカーでの競合回避）
    """
    global _analyzer  # noqa: PLW0603
    # 環境変数でCUDAを無効化（force_cpu=Trueの場合）
    if force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    _analyzer = ImageQualityAnalyzer(genre=genre)
    pid = os.getpid()
    device = _analyzer.device
    logger.info(f"Worker {pid}: ImageQualityAnalyzer initialized (device={device})")


def _analyze_single(path: str) -> Optional[ImageMetrics]:
    """単一の画像を分析する.

    ワーカープロセス内のグローバル_analyzerを使用する。

    Args:
        path: 画像ファイルパス

    Returns:
        分析結果（失敗時はNone）
    """
    global _analyzer
    assert _analyzer is not None, "Analyzer not initialized. Call _init_worker first."
    return _analyzer.analyze(path)


class ImageQualityAnalyzerPool:
    """ImageQualityAnalyzerのmultiprocessing pool.

    各ワーカープロセスがCLIPモデルを保持することで、モデルロードコストを削減する。
    """

    def __init__(
        self,
        genre: str = "mixed",
        num_workers: Optional[int] = None,
        force_cpu: bool = False,
    ):
        """プールを初期化する.

        Args:
            genre: ジャンル重みの種類
            num_workers: ワーカー数（Noneで自動設定）
            force_cpu: GPUを無効化してCPUを強制する
                       （複数ワーカーでのGPUメモリ競合を回避）
        """
        self.genre = genre
        self._pool: Optional[Pool] = None
        self._num_workers = num_workers
        self._force_cpu = force_cpu
        self._actual_workers: int = 0  # 実際のワーカー数をキャッシュ

    def __enter__(self) -> "ImageQualityAnalyzerPool":
        """コンテキストマネージャーとして使用する場合のエントリー."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[Any],
    ) -> Literal[False]:
        """コンテキストマネージャーとして使用する場合のイグジット."""
        self.close()
        return False

    def start(self) -> None:
        """プールを開始する."""
        if self._pool is not None:
            logger.warning("Pool is already started")
            return

        self._pool = Pool(
            processes=self._num_workers,
            initializer=_init_worker,
            initargs=(self.genre, self._force_cpu),
        )
        self._actual_workers = self._num_workers or os.cpu_count() or 1
        logger.info(f"AnalyzerPool started with {self._actual_workers} workers")

    def close(self) -> None:
        """プールを閉じる."""
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None
            logger.info("AnalyzerPool closed")

    def analyze_batch(self, paths: List[str]) -> List[Optional[ImageMetrics]]:
        """複数の画像を並列分析する.

        Args:
            paths: 画像ファイルパスのリスト

        Returns:
            分析結果のリスト（失敗した画像はNone）
        """
        if self._pool is None:
            raise RuntimeError(
                "Pool is not started. Call start() or use context manager"
            )

        logger.info(
            f"Analyzing {len(paths)} images with {self._actual_workers} workers"
        )
        results = self._pool.map(_analyze_single, paths)
        return results

    @property
    def num_workers(self) -> int:
        """ワーカー数を取得する."""
        if self._pool is None:
            raise RuntimeError("Pool is not started")
        return self._actual_workers
