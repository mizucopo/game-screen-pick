"""バッチ処理パイプライン - 複数画像のバッチ処理をオーケストレーションする."""

import concurrent.futures
import logging
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import cv2
import numpy as np
import torch
from PIL import Image

from ..models.analyzer_config import AnalyzerConfig
from ..models.image_metrics import ImageMetrics
from ..utils.exception_handler import ExceptionHandler
from ..utils.image_utils import ImageUtils
from .feature_extractor import FeatureExtractor
from .metric_calculator import MetricCalculator

logger = logging.getLogger(__name__)


class BatchPipeline:
    """バッチ処理パイプライン.

    複数の画像をバッチ処理で解析するオーケストレーションを行う。
    チャンク分割とlookaheadプリロードによるメモリ効率化を実現する。
    """

    # 進捗表示の間隔
    PROGRESS_REPORT_INTERVAL: int = 500

    def __init__(
        self,
        feature_extractor: "FeatureExtractor",
        metric_calculator: "MetricCalculator",
        config: AnalyzerConfig,
    ):
        """バッチ処理パイプラインを初期化する.

        Args:
            feature_extractor: 特徴抽出器
            metric_calculator: メトリクス計算器
            config: アナライザー設定
        """
        self.feature_extractor = feature_extractor
        self.metric_calculator = metric_calculator
        self.config = config
        # OpenCVのスレッド数を1に設定し、ThreadPoolExecutorとの競合を回避
        cv2.setNumThreads(1)
        # 結果構築の並列処理ワーカー数を決定（デフォルトはmin(8, cpu_count-1)）
        if config.result_max_workers is None:
            cpu_count = os.cpu_count() or 1
            self._result_max_workers = min(8, max(1, cpu_count - 1))
        else:
            self._result_max_workers = config.result_max_workers

        # 共有Executor（チャンク間で再利用）
        self._executor: ThreadPoolExecutor | None = None
        # プリロード用Executor（io_max_workersベース、load_and_preprocess_images専用）
        self._preload_executor: ThreadPoolExecutor | None = None
        # 内部処理用Executor（load_and_preprocess_images内部のmap用）
        # io_max_workers=1でデッドロックを回避するため、別poolを使用
        self._io_executor: ThreadPoolExecutor | None = None

    @staticmethod
    def _convert_batch_features_to_numpy(
        clip_features_list: list[torch.Tensor | None],
        batch_features: torch.Tensor | None,
        valid_indices: list[int],
    ) -> list[np.ndarray | None]:
        """セマンティック計算済みのバッチTensorを再利用してCPUに転送.

        二重stackを回避するため、calculate_semantic_score_batchで
        既にstackされたバッチTensorを再利用する。

        Args:
            clip_features_list: 元のCLIP特徴リスト（Noneを含む場合あり）
            batch_features: セマンティック計算済みのGPU上のバッチTensor
            valid_indices: 有効な特徴のインデックスリスト

        Returns:
            CPU上のNumPy配列リスト（元のNoneは保持）
        """
        results: list[np.ndarray | None] = [None] * len(clip_features_list)

        if batch_features is None or not valid_indices:
            return results

        with torch.inference_mode():
            batch_cpu = batch_features.cpu()
            batch_np = batch_cpu.numpy()

        for j, idx in enumerate(valid_indices):
            results[idx] = batch_np[j]

        return results

    def process_batch(
        self,
        paths: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> list[ImageMetrics | None]:
        """複数の画像をバッチ処理で解析する.

        パイプライン構成（各チャンク内で4ステージを並列実行）:
        1. ステージ1: Lookaheadプリロード（I/O+CV前処理）
        2. ステージ2: バッチCLIP推論
        3. ステージ3: バッチセマンティックスコア計算
        4. ステージ4: 並列結果構築（raw metric + feature結合）

        メモリ効率化のためチャンク単位でストリーミング処理:
        - メモリ予算に基づいて動的にチャンクサイズを決定
        - チャンク単位で「前処理→CLIP→結果確定→解放」を実行
        - 大規模画像時のスワップ回避で速度安定化

        パフォーマンス最適化:
        - GPU→CPU転送をチャンク単位でまとめて実行
        - 先読み（lookahead）でI/OとCLIP推論をオーバーラップ

        Args:
            paths: 画像ファイルパスのリスト
            batch_size: CLIP推論のバッチサイズ（デフォルト32）
            show_progress: 進捗表示をするかどうか

        Returns:
            解析結果のリスト（失敗した画像はNone）
        """
        results: list[ImageMetrics | None] = [None] * len(paths)

        # メモリ予算に基づいてチャンク境界を計算
        chunk_boundaries = self._compute_chunk_boundaries(
            paths,
            self.config.max_memory_mb,
            self.config.min_chunk_size,
            self.config.max_dim,
        )

        # 先読み用のスレッドプール（インスタンスで再利用）
        preload_executor = self._get_preload_executor()
        PilImagesFuture = concurrent.futures.Future[list[Image.Image | None]]
        preload_futures: dict[int, PilImagesFuture] = {}

        # 最初のチャンクをプリロード（lookahead=2）
        lookahead = 2
        for i in range(min(lookahead, len(chunk_boundaries))):
            chunk_start, chunk_end = chunk_boundaries[i]
            chunk_paths = paths[chunk_start:chunk_end]
            preload_futures[i] = preload_executor.submit(
                self.load_and_preprocess_images,
                chunk_paths,
                self.config.max_dim,
            )

        for chunk_idx, (chunk_start, chunk_end) in enumerate(chunk_boundaries):
            chunk_paths = paths[chunk_start:chunk_end]
            logger.debug(
                f"チャンク処理開始: {chunk_idx + 1}/{len(chunk_boundaries)} "
                f"({chunk_start}-{chunk_end}, {len(chunk_paths)}枚)"
            )

            # プリロードされた画像を取得
            logger.debug(f"プリロード画像を取得中: チャンク{chunk_idx}")
            pil_images = preload_futures[chunk_idx].result()
            valid_count = len([p for p in pil_images if p is not None])
            logger.debug(f"プリロード完了: {valid_count}枚有効")
            del preload_futures[chunk_idx]

            # 次のチャンクをプリロード
            next_idx = chunk_idx + lookahead
            if next_idx < len(chunk_boundaries) and next_idx not in preload_futures:
                next_start, next_end = chunk_boundaries[next_idx]
                next_paths = paths[next_start:next_end]
                preload_futures[next_idx] = preload_executor.submit(
                    self.load_and_preprocess_images,
                    next_paths,
                    self.config.max_dim,
                )
                logger.debug(f"次チャンクのプリロード開始: チャンク{next_idx}")

            # ステージ2: チャンク単位でバッチCLIP推論を実行
            logger.debug(f"CLIP推論開始: {len(pil_images)}枚")
            clip_features_list = self.feature_extractor.extract_clip_features_batch(
                pil_images, initial_batch_size=batch_size
            )
            logger.debug(
                f"CLIP推論完了: "
                f"{len([f for f in clip_features_list if f is not None])}枚成功"
            )

            # ステージ3: セマンティックスコアをバッチ計算（バッチTensorを再利用）
            logger.debug("セマンティックスコア計算開始")
            semantic_scores, batch_features, valid_indices = (
                self.metric_calculator.calculate_semantic_score_batch(
                    clip_features_list
                )
            )
            logger.debug(f"セマンティックスコア計算完了: {len(valid_indices)}枚有効")

            # ステージ3.5: バッチTensorを再利用してGPU→CPU転送
            clip_features_np_list = self._convert_batch_features_to_numpy(
                clip_features_list, batch_features, valid_indices
            )

            # ステージ4: チャンク単位で結果を構築（並列化）
            chunk_results = self._process_result_parallel(
                chunk_paths=chunk_paths,
                pil_images=pil_images,
                clip_features_list=clip_features_np_list,
                semantic_scores=semantic_scores,
                chunk_start=chunk_start,
                total_paths=len(paths),
                show_progress=show_progress,
            )

            # 結果を正しい位置にマッピング
            for i, chunk_result in enumerate(chunk_results):
                results[chunk_start + i] = chunk_result

            # チャンク処理完了後にメモリを解放
            del pil_images, clip_features_list, clip_features_np_list

        return results

    @staticmethod
    def _compute_chunk_boundaries(
        paths: list[str],
        max_memory_mb: int,
        min_chunk_size: int,
        max_dim: int | None = None,
    ) -> list[tuple[int, int]]:
        """メモリ予算に基づいてチャンク境界を計算する.

        画像解像度ベースでメモリ使用量を見積もり、max_dimで縮小後の
        サイズを考慮してチャンクを分割する。

        見積もり式:
        - 1ピクセルあたり約4バイト（RGBA + PILオーバーヘッド）
        - PIL Image: width × height × 4 bytes
        - CLIP Tensor: 追加で同等程度
        - 安全係数: 3.0

        Args:
            paths: 画像ファイルパスのリスト
            max_memory_mb: チャンクあたりの最大メモリ予算（MB）
            min_chunk_size: 最低限確保するチャンクサイズ
            max_dim: 長辺の最大ピクセル数（Noneの場合は縮小しない）

        Returns:
            (start_index, end_index) のタプルリスト
        """
        BYTES_PER_PIXEL = 4  # RGBA
        SAFETY_FACTOR = 3.0  # CLIP Tensor等を考慮した安全係数
        DEFAULT_MEMORY = 1024 * 1024  # 読み込み失敗時のデフォルト値（1MB相当）

        max_memory_bytes = max_memory_mb * 1024 * 1024
        chunks = []
        current_start = 0
        current_memory = 0
        current_count = 0

        for i, path in enumerate(paths):
            # 画像の解像度からメモリを見積もる
            try:
                with Image.open(path) as img:
                    width, height = img.size
                    # max_dimで縮小後のサイズを計算
                    if max_dim is not None:
                        scale = min(max_dim / max(width, height), 1.0)
                        width = int(width * scale)
                        height = int(height * scale)
                    estimated_memory = int(
                        width * height * BYTES_PER_PIXEL * SAFETY_FACTOR
                    )
            except Exception:
                # 読み込み失敗時はデフォルト値（1MB相当）
                estimated_memory = DEFAULT_MEMORY

            # チャンク追加でメモリ予算を超える場合は新規チャンクを検討
            would_exceed = current_memory + estimated_memory > max_memory_bytes
            has_min_images = current_count >= min_chunk_size
            can_split = i > 0

            if would_exceed and has_min_images and can_split:
                chunks.append((current_start, i))
                current_start = i
                current_memory = estimated_memory
                current_count = 1
            else:
                current_memory += estimated_memory
                current_count += 1

        # 最後のチャンクを追加
        if current_start < len(paths):
            final_chunk = (current_start, len(paths))
            if chunks and final_chunk[1] - final_chunk[0] < min_chunk_size:
                prev_start, _ = chunks.pop()
                chunks.append((prev_start, final_chunk[1]))
            else:
                chunks.append(final_chunk)

        return chunks

    def load_and_preprocess_images(
        self,
        paths: list[str],
        max_dim: int | None = None,
    ) -> list[Image.Image | None]:
        """複数のパスからPIL画像を読み込み、前処理まで並列実行する.

        I/O（画像読み込み）とCPU-bound処理（RGB変換・縮小）をThreadPoolExecutorで並列化.
        内部処理用io_executorを使用し、プリロード用executorとの競合を回避する。

        Args:
            paths: 画像ファイルパスのリスト
            max_dim: 長辺の最大ピクセル数（Noneの場合は縮小しない）

        Returns:
            PIL画像のリスト（失敗したパスはNone）
        """
        if max_dim is not None:
            load_func: Callable[[str], Image.Image | None] = partial(
                ImageUtils.load_as_rgb_resized, max_dim=max_dim
            )
        else:
            load_func = ImageUtils.load_as_rgb

        # io_executorを使用（preload_executorとのデッドロック回避）
        executor = self._get_io_executor()
        results = list(executor.map(load_func, paths))

        return results

    def _process_single_result(
        self,
        path: str,
        pil_img: Image.Image,
        clip_features: np.ndarray,
        semantic: float,
    ) -> ImageMetrics | None:
        """結果構築の単一画像処理.

        raw metric計算、feature結合、総合スコア計算を行う。

        Args:
            path: 画像ファイルパス
            pil_img: PIL画像（既にmax_dim以下に縮小済み）
            clip_features: CLIP特徴（np.ndarray、CPU上）
            semantic: セマンティックスコア

        Returns:
            ImageMetricsオブジェクト（処理失敗時はNone）
        """
        try:
            # pil_imgは既にload_and_preprocess_imagesで縮小済み
            img = ImageUtils.pil_to_cv2(pil_img)

            # 生メトリクスと正規化メトリクスのみ計算
            # （セマンティックスコアはバッチ計算済みの値を使用）
            raw, norm = self.metric_calculator.calculate_raw_norm_metrics(img)
            # バッチ計算したセマンティックスコアを使用して総合スコアを計算
            total = self.metric_calculator.calculate_total_score(raw, norm, semantic)

            # HSV特徴とCLIP特徴を結合
            features = self.feature_extractor.extract_combined_features(
                img,
                clip_features,
            )

            return ImageMetrics(path, raw, norm, semantic, total, features)
        except ExceptionHandler.get_expected_image_errors() as e:
            logger.warning(
                f"画像分析をスキップしました: {path}, 理由: {type(e).__name__}: {e}"
            )
            return None

    def _process_result_parallel(
        self,
        chunk_paths: list[str],
        pil_images: list[Image.Image | None],
        clip_features_list: list[np.ndarray | None],
        semantic_scores: list[float | None],
        chunk_start: int,
        total_paths: int,
        show_progress: bool,
    ) -> list[ImageMetrics | None]:
        """結果構築（raw metric + feature結合）を並列処理.

        ThreadPoolExecutorを使用してチャンク内の画像処理を並列化する。

        Args:
            chunk_paths: チャンク内の画像パスリスト
            pil_images: PIL画像リスト
            clip_features_list: CLIP特徴リスト（np.ndarray、CPU上）
            semantic_scores: セマンティックスコアリスト
            chunk_start: チャンクの開始インデックス
            total_paths: 総画像数
            show_progress: 進捗表示フラグ

        Returns:
            ImageMetricsオブジェクトのリスト（失敗時はNone）
        """
        # タスクの型エイリアス
        # TaskData: (path, pil_img, clip_features, semantic, global_idx) または None
        # TaskTuple: (local_idx, TaskData) - local_idxで結果の順序を維持
        TaskData = tuple[str, Image.Image, np.ndarray, float, int] | None
        TaskTuple = tuple[int, TaskData]

        # 並列処理するタスクを準備
        tasks: list[TaskTuple] = []
        zipped = zip(
            chunk_paths,
            pil_images,
            clip_features_list,
            semantic_scores,
            strict=True,
        )
        for i, (path, pil_img, clip_features, semantic) in enumerate(zipped):
            if pil_img is None or clip_features is None or semantic is None:
                tasks.append((i, None))  # Noneは処理スキップを示す
            else:
                # タスクを保存（インデックス付きで順序維持）
                global_idx = chunk_start + i
                tasks.append((i, (path, pil_img, clip_features, semantic, global_idx)))

        # 並列処理関数
        def process_task(task_info: TaskTuple) -> tuple[int, ImageMetrics | None]:
            """単一タスクを処理する."""
            idx, data = task_info
            if data is None:
                return idx, None
            path, pil_img, clip_features, semantic, global_idx = data
            interval = BatchPipeline.PROGRESS_REPORT_INTERVAL
            if show_progress and global_idx % interval == 0:
                logger.info(f"解析済み: {global_idx}/{total_paths}")
            result = self._process_single_result(path, pil_img, clip_features, semantic)
            return idx, result

        # 共有Executorを使用
        executor = self._get_executor()
        executor_results = list(executor.map(process_task, tasks))

        # 結果を配置
        results: list[ImageMetrics | None] = [None] * len(tasks)
        for idx, result in executor_results:
            results[idx] = result

        return results

    def _get_executor(self) -> ThreadPoolExecutor:
        """スレッドプールを取得または作成.

        Returns:
            ThreadPoolExecutorインスタンス
        """
        if self._executor is None:
            # ThreadPoolExecutor(max_workers=0) は ValueError になるため
            # 0 の場合は 1 (シングルスレッド) に変換
            workers = self._result_max_workers if self._result_max_workers > 0 else 1
            self._executor = ThreadPoolExecutor(max_workers=workers)
        return self._executor

    def _get_preload_executor(self) -> ThreadPoolExecutor:
        """プリロード用スレッドプールを取得または作成.

        load_and_preprocess_images の実行専用プール。
        io_max_workers=1 でのデッドロック回避のため、_get_io_executor とは
        別々のインスタンスを使用する。

        Returns:
            ThreadPoolExecutorインスタンス
        """
        if self._preload_executor is None:
            # プリロードは並行度を必要とするため、io_max_workersが1の場合は2を確保
            max_workers = self.config.io_max_workers or 1
            if max_workers < 2:
                max_workers = 2
            self._preload_executor = ThreadPoolExecutor(max_workers=max_workers)
        return self._preload_executor

    def _get_io_executor(self) -> ThreadPoolExecutor:
        """内部I/O処理用スレッドプールを取得または作成.

        load_and_preprocess_images 内部の map 処理用。
        プリロードexecutor とは分離し、io_max_workers=1 でのデッドロックを回避。

        Returns:
            ThreadPoolExecutorインスタンス
        """
        if self._io_executor is None:
            max_workers = self.config.io_max_workers or 1
            self._io_executor = ThreadPoolExecutor(max_workers=max_workers)
        return self._io_executor

    def close(self) -> None:
        """スレッドプールをクリーンアップ."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        if self._preload_executor is not None:
            self._preload_executor.shutdown(wait=True)
            self._preload_executor = None
        if self._io_executor is not None:
            self._io_executor.shutdown(wait=True)
            self._io_executor = None

    def __enter__(self) -> "BatchPipeline":
        """コンテキストマネージャーのエントリー.

        Returns:
            自分自身
        """
        return self

    def __exit__(self, *args: object) -> None:
        """コンテキストマネージーの exit.

        Args:
            *args: 例外情報（unused）
        """
        self.close()
