"""Worker class for ImageQualityAnalyzer in multiprocessing.

各ワーカープロセスがモデルを保持することで、モデルロードコストを削減する。
"""

import logging
import os
from typing import Optional

from .image_quality_analyzer import ImageQualityAnalyzer
from ..models.image_metrics import ImageMetrics

logger = logging.getLogger(__name__)


class AnalyzerWorker:
    """ワーカープロセスでImageQualityAnalyzerを管理するクラス."""

    # クラス変数: ワーカープロセスごとにAnalyzerWorkerインスタンスを保持
    _worker: Optional["AnalyzerWorker"] = None

    def __init__(self, genre: str, force_cpu: bool) -> None:
        """ワーカーを初期化.

        Args:
            genre: ジャンル重みの種類
            force_cpu: CPUを強制するかどうか
        """
        device = "cpu" if force_cpu else None
        self.analyzer = ImageQualityAnalyzer(genre=genre, device=device)
        pid = os.getpid()
        logger.info(
            f"Worker {pid}: ImageQualityAnalyzer initialized "
            f"(device={self.analyzer.device})"
        )

    def analyze(self, path: str) -> Optional[ImageMetrics]:
        """単一の画像を分析.

        Args:
            path: 画像ファイルパス

        Returns:
            分析結果（失敗時はNone）
        """
        return self.analyzer.analyze(path)

    @staticmethod
    def init_worker(genre: str = "mixed", force_cpu: bool = False) -> None:
        """ワーカープロセスの初期化関数.

        各ワーカープロセスで1回だけ呼び出され、AnalyzerWorkerを初期化する。

        Args:
            genre: ジャンル重みの種類
            force_cpu: GPUを無効化してCPUを強制する（複数ワーカーでの競合回避）
        """
        AnalyzerWorker._worker = AnalyzerWorker(genre, force_cpu)

    @staticmethod
    def analyze_single(path: str) -> Optional[ImageMetrics]:
        """単一の画像を分析する.

        ワーカープロセス内のAnalyzerWorker._workerを使用する。

        Args:
            path: 画像ファイルパス

        Returns:
            分析結果（失敗時はNone）
        """
        worker = AnalyzerWorker._worker
        if worker is None:
            raise RuntimeError(
                "AnalyzerWorker not initialized. Call init_worker first."
            )
        return worker.analyze(path)
