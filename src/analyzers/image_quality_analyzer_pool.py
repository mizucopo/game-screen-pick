"""Multiprocessing pool for ImageQualityAnalyzer with model persistence.

各ワーカープロセスがモデルを保持することで、モデルロードコストを削減する。
"""

import logging
import os
from multiprocessing.pool import Pool
from typing import Any, Literal

from ..models.image_metrics import ImageMetrics
from .analyzer_worker import AnalyzerWorker

logger = logging.getLogger(__name__)


class ImageQualityAnalyzerPool:
    """ImageQualityAnalyzerのmultiprocessing pool.

    各ワーカープロセスがCLIPモデルを保持することで、モデルロードコストを削減する。
    """

    def __init__(
        self,
        num_workers: int | None = None,
        force_cpu: bool = False,
    ):
        """プールを初期化する.

        Args:
            num_workers: ワーカー数（Noneで自動設定）
            force_cpu: GPUを無効化してCPUを強制する
                       （複数ワーカーでのGPUメモリ競合を回避）
        """
        self._pool: Pool | None = None
        self._num_workers = num_workers
        self._force_cpu = force_cpu
        self._actual_workers: int = 0  # 実際のワーカー数をキャッシュ

    def __enter__(self) -> "ImageQualityAnalyzerPool":
        """コンテキストマネージャーとして使用する場合のエントリー."""
        self.start()
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: Any,
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
            initializer=AnalyzerWorker.init_worker,
            initargs=(self._force_cpu,),
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

    def analyze_batch(self, paths: list[str]) -> list[ImageMetrics | None]:
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
        results = self._pool.map(AnalyzerWorker.analyze_single, paths)
        return results

    @property
    def num_workers(self) -> int:
        """ワーカー数を取得する."""
        if self._pool is None:
            raise RuntimeError("Pool is not started")
        return self._actual_workers
