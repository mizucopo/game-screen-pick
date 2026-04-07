"""バッチ処理パイプライン.

中立解析に必要なI/O、CLIP推論前後のデータ変換、結果構築を
チャンク単位で制御し、メモリ予算内で大量画像を処理する。
"""

import concurrent.futures
import logging
import os
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
import torch
from PIL import Image

from ..models.analyzed_image import AnalyzedImage
from ..models.analyzer_config import AnalyzerConfig
from ..utils.exception_handler import ExceptionHandler
from ..utils.image_utils import ImageUtils
from .feature_extractor import FeatureExtractor
from .layout_analyzer import LayoutAnalyzer
from .metric_calculator import MetricCalculator

logger = logging.getLogger(__name__)

type PilImagesFuture = concurrent.futures.Future[list[Image.Image | None]]
type TaskData = tuple[str, Image.Image, np.ndarray, int] | None
type TaskTuple = tuple[int, TaskData]


class BatchPipeline:
    """複数画像の中立解析をオーケストレーションする.

    チャンク分割とlookaheadプリロードを組み合わせ、
    画像読み込み、CLIP特徴抽出、特徴変換、結果構築を
    一貫した順序で流す役割を持つ。
    """

    PROGRESS_REPORT_INTERVAL: int = 500

    def __init__(
        self,
        feature_extractor: "FeatureExtractor",
        metric_calculator: "MetricCalculator",
        config: AnalyzerConfig,
    ):
        """バッチ処理パイプラインを初期化する.

        Args:
            feature_extractor: CLIP特徴と結合特徴を抽出するコンポーネント。
            metric_calculator: 生メトリクスと正規化メトリクスを計算する器。
            config: チャンクサイズ、リサイズ、並列数などを含む設定。
        """
        self.feature_extractor = feature_extractor
        self.metric_calculator = metric_calculator
        self.config = config
        if config.result_max_workers is None:
            cpu_count = os.cpu_count() or 1
            self._result_max_workers = min(8, max(1, cpu_count - 1))
        else:
            self._result_max_workers = config.result_max_workers

        self._executor: ThreadPoolExecutor | None = None
        self._preload_executor: ThreadPoolExecutor | None = None
        self._io_executor: ThreadPoolExecutor | None = None
        self._executor_lock = threading.Lock()
        self._preload_lock = threading.Lock()
        self._io_lock = threading.Lock()

    @staticmethod
    def _convert_batch_features_to_numpy(
        clip_features_list: list[torch.Tensor | None],
    ) -> list[np.ndarray | None]:
        """CLIP特徴をCPU上のNumPy配列へ変換する.

        バッチ推論の結果は失敗要素を `None` として含むため、
        有効なTensorだけをまとめて `torch.stack` し、
        CPUへ転送した後で元のインデックス順に戻す。
        後段の特徴結合や類似度計算がNumPy前提のため、
        ここで一括変換しておく。

        Args:
            clip_features_list: バッチ推論から得たCLIP特徴のリスト。
                推論に失敗した要素は `None` を取る。

        Returns:
            元の順序を維持したNumPy配列のリスト。
            失敗した要素は `None` のまま残す。
        """
        results: list[np.ndarray | None] = [None] * len(clip_features_list)
        valid_indices = [
            idx
            for idx, features in enumerate(clip_features_list)
            if features is not None
        ]
        if not valid_indices:
            return results

        valid_tensors: list[torch.Tensor] = []
        for idx in valid_indices:
            tensor = clip_features_list[idx]
            if tensor is not None:
                valid_tensors.append(tensor)
        batch_np = torch.stack(valid_tensors).cpu().numpy()
        for array_index, result_index in enumerate(valid_indices):
            results[result_index] = batch_np[array_index]
        return results

    def process_batch(
        self,
        paths: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> list[AnalyzedImage | None]:
        """複数の画像をバッチ処理で解析する.

        現在のパイプラインはチャンク単位で以下を行う。
        1. lookahead付きで画像をプリロードする
        2. CLIP特徴をバッチ推論する
        3. 特徴をCPU上のNumPy配列へまとめて変換する
        4. 生メトリクス、正規化メトリクス、結合特徴、レイアウト推定を
           並列に組み立てて `AnalyzedImage` にまとめる

        チャンク境界はメモリ予算から動的に決まり、
        大量画像でもスワップを起こしにくいサイズへ分割される。
        また、次チャンクを先読みすることでI/Oと推論の待ち時間を重ねる。

        Args:
            paths: 解析対象の画像ファイルパス一覧。
            batch_size: CLIP推論時に使う初期バッチサイズ。
            show_progress: 一定間隔で進捗ログを出すかどうか。

        Returns:
            入力順に対応した解析結果のリスト。
            読み込みや解析に失敗した画像は `None` になる。
        """
        results: list[AnalyzedImage | None] = [None] * len(paths)
        chunk_boundaries = self._compute_chunk_boundaries(
            paths,
            self.config.max_memory_gb,
            self.config.min_chunk_size,
            self.config.max_dim,
        )

        preload_executor = self._get_preload_executor()
        preload_futures: dict[int, PilImagesFuture] = {}

        lookahead = 2
        for chunk_idx in range(min(lookahead, len(chunk_boundaries))):
            chunk_start, chunk_end = chunk_boundaries[chunk_idx]
            chunk_paths = paths[chunk_start:chunk_end]
            preload_futures[chunk_idx] = preload_executor.submit(
                self.load_and_preprocess_images,
                chunk_paths,
                self.config.max_dim,
            )

        for chunk_idx, (chunk_start, chunk_end) in enumerate(chunk_boundaries):
            chunk_paths = paths[chunk_start:chunk_end]
            pil_images = preload_futures[chunk_idx].result()
            del preload_futures[chunk_idx]

            self._preload_next_chunks(
                paths, chunk_boundaries, chunk_idx, lookahead, preload_futures
            )

            clip_features_list = self.feature_extractor.extract_clip_features_batch(
                pil_images,
                initial_batch_size=batch_size,
            )
            clip_features_np_list = self._convert_batch_features_to_numpy(
                clip_features_list
            )

            chunk_results = self._process_result_parallel(
                chunk_paths=chunk_paths,
                pil_images=pil_images,
                clip_features_list=clip_features_np_list,
                chunk_start=chunk_start,
                total_paths=len(paths),
                show_progress=show_progress,
            )
            for offset, chunk_result in enumerate(chunk_results):
                results[chunk_start + offset] = chunk_result

            # チャンク完了時の進捗ログ
            if show_progress:
                logger.info(f"処理済み: {chunk_end}/{len(paths)}")

        return results

    def _preload_next_chunks(
        self,
        paths: list[str],
        chunk_boundaries: list[tuple[int, int]],
        chunk_idx: int,
        lookahead: int,
        preload_futures: dict[int, PilImagesFuture],
    ) -> None:
        """先読みチャンクをプリロードExecutorに投入する."""
        next_idx = chunk_idx + lookahead
        if next_idx < len(chunk_boundaries) and next_idx not in preload_futures:
            next_start, next_end = chunk_boundaries[next_idx]
            next_paths = paths[next_start:next_end]
            preload_futures[next_idx] = self._get_preload_executor().submit(
                self.load_and_preprocess_images,
                next_paths,
                self.config.max_dim,
            )

    @staticmethod
    def _compute_chunk_boundaries(
        paths: list[str],
        max_memory_gb: int,
        min_chunk_size: int,
        max_dim: int | None = None,
    ) -> list[tuple[int, int]]:
        """メモリ予算に基づいてチャンク境界を計算する.

        画像解像度から概算メモリ使用量を見積もり、
        `max_memory_gb` を超えない範囲でチャンクを切り出す。
        `max_dim` が設定されている場合は、リサイズ後の想定解像度を使って
        見積もるため、実行時のメモリ挙動に近い境界を得られる。

        Args:
            paths: 入力画像パスのリスト。
            max_memory_gb: 1チャンクあたりに使ってよい最大メモリ量。
            min_chunk_size: 分割しすぎを避けるための最小チャンクサイズ。
            max_dim: 長辺の最大ピクセル数。 `None` の場合は縮小を考慮しない。

        Returns:
            `(start_index, end_index)` 形式のチャンク境界リスト。
        """
        bytes_per_pixel = 4
        safety_factor = 3.0
        default_memory = 1024 * 1024

        max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
        chunks: list[tuple[int, int]] = []
        current_start = 0
        current_memory = 0
        current_count = 0

        for idx, path in enumerate(paths):
            try:
                with Image.open(path) as img:
                    width, height = img.size
                    if max_dim is not None:
                        scale = min(max_dim / max(width, height), 1.0)
                        width = int(width * scale)
                        height = int(height * scale)
                    estimated_memory = int(
                        width * height * bytes_per_pixel * safety_factor
                    )
            except ExceptionHandler.get_expected_image_errors():
                estimated_memory = default_memory

            would_exceed = current_memory + estimated_memory > max_memory_bytes
            has_min_images = current_count >= min_chunk_size
            can_split = idx > 0

            if would_exceed and has_min_images and can_split:
                chunks.append((current_start, idx))
                current_start = idx
                current_memory = estimated_memory
                current_count = 1
            else:
                current_memory += estimated_memory
                current_count += 1

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
        """複数画像を並列に読み込み、前処理済みPIL画像へ揃える.

        読み込みとRGB変換、必要に応じたリサイズをI/O向けExecutorで処理する。
        プリロード用Executorとは分離し、`io_max_workers=1` のときでも
        デッドロックしにくい構成を保つ。

        Args:
            paths: 読み込む画像パスのリスト。
            max_dim: 長辺の最大ピクセル数。指定時は縮小版を返す。

        Returns:
            PIL画像のリスト。読み込みに失敗した要素は `None` になる。
        """
        if max_dim is not None:
            load_func: Callable[[str], Image.Image | None] = partial(
                ImageUtils.load_as_rgb_resized,
                max_dim=max_dim,
            )
        else:
            load_func = ImageUtils.load_as_rgb

        executor = self._get_io_executor()
        return list(executor.map(load_func, paths))

    def _process_single_result(
        self,
        path: str,
        pil_img: Image.Image,
        clip_features: np.ndarray,
    ) -> AnalyzedImage | None:
        """単一画像の解析結果を構築する.

        OpenCV形式への変換後に、画質メトリクス、結合特徴、
        レイアウトヒューリスティクスを計算し、
        scene判定前の中立データモデル `AnalyzedImage` にまとめる。

        Args:
            path: 元画像のパス。
            pil_img: 読み込みと前処理が済んだPIL画像。
            clip_features: CPU上へ移したCLIP特徴ベクトル。

        Returns:
            構築済みの `AnalyzedImage` 。想定内の画像エラー時は `None` 。
        """
        try:
            img = ImageUtils.pil_to_cv2(pil_img)
            raw_metrics, normalized_metrics = (
                self.metric_calculator.calculate_all_metrics(img)
            )
            hsv_features = self.feature_extractor.extract_hsv_features(img)
            combined_features = self.feature_extractor.extract_combined_features(
                img,
                clip_features,
                hsv_features=hsv_features,
            )
            content_features = self.feature_extractor.extract_content_features(
                img,
                raw_metrics,
                hsv_features=hsv_features,
            )
            layout_heuristics = LayoutAnalyzer.analyze(img)
            return AnalyzedImage(
                path=path,
                raw_metrics=raw_metrics,
                normalized_metrics=normalized_metrics,
                clip_features=clip_features,
                combined_features=combined_features,
                content_features=content_features,
                layout_heuristics=layout_heuristics,
            )
        except ExceptionHandler.get_expected_image_errors() as error:
            logger.warning(
                "画像分析をスキップしました: "
                f"{path}, 理由: {type(error).__name__}: {error}"
            )
            return None

    def _process_result_parallel(
        self,
        chunk_paths: list[str],
        pil_images: list[Image.Image | None],
        clip_features_list: list[np.ndarray | None],
        chunk_start: int,
        total_paths: int,
        show_progress: bool,
    ) -> list[AnalyzedImage | None]:
        """結果構築を並列処理する.

        1チャンク分のPIL画像とCLIP特徴を受け取り、
        `AnalyzedImage` の生成だけをスレッドプールで並列化する。
        ローカルインデックスを保持しているため、並列化しても
        返却順序は入力順のまま維持される。

        Args:
            chunk_paths: 現在処理中チャンクの画像パス。
            pil_images: 事前読み込み済みのPIL画像。
            clip_features_list: CPU上へ変換済みのCLIP特徴。
            chunk_start: 全体入力におけるチャンク開始位置。
            total_paths: 進捗表示に使う総画像数。
            show_progress: 一定件数ごとにログを出すかどうか。

        Returns:
            チャンク内入力順に対応した解析結果リスト。
        """
        tasks: list[TaskTuple] = []
        zipped = zip(chunk_paths, pil_images, clip_features_list, strict=True)
        for local_index, (path, pil_img, clip_features) in enumerate(zipped):
            if pil_img is None or clip_features is None:
                tasks.append((local_index, None))
            else:
                global_index = chunk_start + local_index
                tasks.append(
                    (local_index, (path, pil_img, clip_features, global_index))
                )

        def process_task(task_info: TaskTuple) -> tuple[int, AnalyzedImage | None]:
            index, data = task_info
            if data is None:
                return index, None
            path, pil_img, clip_features, global_index = data
            if show_progress and global_index % self.PROGRESS_REPORT_INTERVAL == 0:
                logger.info(f"解析済み: {global_index}/{total_paths}")
            return index, self._process_single_result(path, pil_img, clip_features)

        executor = self._get_executor()
        executor_results = list(executor.map(process_task, tasks))

        results: list[AnalyzedImage | None] = [None] * len(tasks)
        for index, result in executor_results:
            results[index] = result
        return results

    def _get_executor(self) -> ThreadPoolExecutor:
        """結果構築用スレッドプールを取得する.

        チャンクごとに作り直さずインスタンス内で再利用し、
        解析結果構築のオーバーヘッドを抑える。

        Returns:
            結果構築用の `ThreadPoolExecutor` 。
        """
        if self._executor is None:
            with self._executor_lock:
                if self._executor is None:
                    workers = (
                        self._result_max_workers if self._result_max_workers > 0 else 1
                    )
                    self._executor = ThreadPoolExecutor(max_workers=workers)
        return self._executor

    def _get_preload_executor(self) -> ThreadPoolExecutor:
        """プリロード用スレッドプールを取得する.

        lookahead読み込み専用のExecutorで、結果構築用と分離して保持する。
        最低2ワーカーを確保し、次チャンクの先読みを進めやすくする。

        Returns:
            プリロード処理用の `ThreadPoolExecutor` 。
        """
        if self._preload_executor is None:
            with self._preload_lock:
                if self._preload_executor is None:
                    max_workers = self.config.io_max_workers or 1
                    if max_workers < 2:
                        max_workers = 2
                    self._preload_executor = ThreadPoolExecutor(max_workers=max_workers)
        return self._preload_executor

    def _get_io_executor(self) -> ThreadPoolExecutor:
        """内部I/O用スレッドプールを取得する.

        `load_and_preprocess_images` の内部でのみ使うExecutorを返し、
        preload側とI/O処理の待ち合わせが競合しないようにする。

        Returns:
            画像読み込み専用の `ThreadPoolExecutor` 。
        """
        if self._io_executor is None:
            with self._io_lock:
                if self._io_executor is None:
                    max_workers = self.config.io_max_workers or 1
                    self._io_executor = ThreadPoolExecutor(max_workers=max_workers)
        return self._io_executor

    def close(self) -> None:
        """保持中のスレッドプールを明示的に破棄する.

        再利用していた各Executorを安全に停止し、
        再度使う場合は次回アクセス時に作り直せる状態へ戻す。
        """
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
        """コンテキストマネージャーに入る.

        Returns:
            この `BatchPipeline` インスタンス自身。
        """
        return self

    def __exit__(self, *args: object) -> None:
        """コンテキストマネージャーを抜ける.

        Args:
            *args: コンテキストマネージャープロトコルが渡す例外情報。
                値にかかわらず内部Executorの解放を行う。
        """
        self.close()
