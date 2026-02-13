"""バッチ処理パイプライン - 複数画像のバッチ処理をオーケストレーションする."""

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, List, Optional, cast

import cv2
import numpy as np
import torch
from PIL import Image

from ..cache.feature_cache import FeatureCache
from ..models.analyzer_config import AnalyzerConfig
from ..models.cache_entry_info import CacheEntryInfo
from ..models.image_metrics import ImageMetrics
from ..models.path_metadata import PathMetadata
from ..utils.exception_handler import ExceptionHandler
from ..utils.image_utils import ImageUtils
from .feature_extractor import FeatureExtractor
from .metric_calculator import MetricCalculator

logger = logging.getLogger(__name__)


class BatchPipeline:
    """バッチ処理パイプライン.

    複数の画像をバッチ処理で解析するオーケストレーションを行う。
    """

    # ファイルサイズからメモリ使用量を見積もるための係数
    MEMORY_ESTIMATION_FACTOR: float = 2.5

    def __init__(
        self,
        feature_extractor: "FeatureExtractor",
        metric_calculator: "MetricCalculator",
        config: AnalyzerConfig,
        cache: FeatureCache | None = None,
        model_name: str = "openai/clip-vit-base-patch32",
        target_text: str = "epic game scenery",
    ):
        """バッチ処理パイプラインを初期化する.

        Args:
            feature_extractor: 特徴抽出器
            metric_calculator: メトリクス計算器
            config: アナライザー設定
            cache: 特徴量キャッシュ（Noneの場合はキャッシュ無効）
            model_name: CLIPモデル名（キャッシュキー用）
            target_text: ターゲットテキスト（キャッシュキー用）
        """
        self.feature_extractor = feature_extractor
        self.metric_calculator = metric_calculator
        self.config = config
        self.cache = cache
        self.model_name = model_name
        self.target_text = target_text
        # OpenCVのスレッド数を1に設定し、ThreadPoolExecutorとの競合を回避
        cv2.setNumThreads(1)
        # 結果構築の並列処理ワーカー数を決定（デフォルトはmin(8, cpu_count-1)）
        if config.result_max_workers is None:
            cpu_count = os.cpu_count() or 1
            self._result_max_workers = min(8, max(1, cpu_count - 1))
        else:
            self._result_max_workers = config.result_max_workers

    @staticmethod
    def _batch_convert_clip_features_to_numpy(
        clip_features_list: List[Optional[torch.Tensor]],
    ) -> List[Optional[np.ndarray]]:
        """CLIP特徴をチャンク単位でまとめてCPUに転送.

        GPU同期コストを削減するため、個別転送の代わりにバッチ転送を使用。

        Args:
            clip_features_list: GPU上のCLIP特徴リスト（Noneを含む場合あり）

        Returns:
            CPU上のNumPy配列リスト（元のNoneは保持）
        """
        valid_indices = [
            i for i, features in enumerate(clip_features_list) if features is not None
        ]

        if not valid_indices:
            return [None] * len(clip_features_list)

        with torch.inference_mode():
            # valid_indicesでフィルタリング済みなのでNoneは含まれない
            valid_tensors = [
                cast(torch.Tensor, clip_features_list[i]) for i in valid_indices
            ]
            batch_tensor = torch.stack(valid_tensors)
            batch_cpu = batch_tensor.cpu()
            batch_np = batch_cpu.numpy()

        results: List[Optional[np.ndarray]] = [None] * len(clip_features_list)
        for j, idx in enumerate(valid_indices):
            results[idx] = batch_np[j]

        return results

    def _get_cached_results(
        self, paths: List[str]
    ) -> tuple[List[Optional[ImageMetrics]], List[PathMetadata]]:
        """キャッシュから取得できる結果を返す.

        パフォーマンス最適化のため、キャッシュヒット時のセマンティックスコア計算は
        バッチ処理で行う。また、キャッシュ一括取得（get_many）でDBアクセスを削減。
        さらに、resolve/statの結果をPathMetadataとして保持し、後続処理での重複を回避。

        Args:
            paths: 画像ファイルパスのリスト

        Returns:
            (キャッシュ結果リスト, 全パスのメタ情報) のタプル
            キャッシュ結果リストは、キャッシュにないインデックスはNone
        """
        if not self.cache:
            # キャッシュ無効時はメタ情報なしですべて未キャッシュとして扱う
            metadata_list = [PathMetadata(path=path) for path in paths]
            return [None] * len(paths), metadata_list

        # 第1パス: 全パスのキャッシュキーとメタ情報を生成
        cache_keys_with_meta = []
        all_metadata: List[PathMetadata] = []
        uncached_metadata: List[PathMetadata] = []

        for i, path in enumerate(paths):
            try:
                absolute_path = os.path.abspath(path)
                file_stat = Path(path).stat()
                cache_key = self.cache.generate_cache_key(
                    absolute_path=absolute_path,
                    file_size=file_stat.st_size,
                    mtime_ns=int(file_stat.st_mtime_ns),
                    model_name=self.model_name,
                    target_text=self.target_text,
                    max_dim=self.config.max_dim,
                )
                # メタ情報を保存
                metadata = PathMetadata(
                    path=path,
                    absolute_path=absolute_path,
                    file_stat=file_stat,
                    cache_key=cache_key,
                )
                all_metadata.append(metadata)
                # インデックスとキーを紐付けて保存
                cache_keys_with_meta.append((i, cache_key, path))
            except (OSError, ValueError):
                # ファイルアクセスエラー等は未キャッシュとして扱う
                metadata = PathMetadata(path=path)
                all_metadata.append(metadata)
                uncached_metadata.append(metadata)

        # get_manyで一括取得
        keys_to_lookup = [meta[1] for meta in cache_keys_with_meta]
        cached_entries: List[Optional[CacheEntryInfo]] = [None] * len(paths)

        if keys_to_lookup:
            cache_results = self.cache.get_many(keys_to_lookup)

            # 結果をマッピング（タプルを直接使用して文字列化を回避）
            for idx, cache_key, path in cache_keys_with_meta:
                key_id = (
                    cache_key["absolute_path"],
                    cache_key["file_size"],
                    cache_key["mtime_ns"],
                    cache_key["model_name"],
                    cache_key["target_text"],
                    cache_key["max_dim"],
                    cache_key["metrics_version"],
                )
                entry = cache_results.get(key_id)
                if entry is not None:
                    cached_entries[idx] = CacheEntryInfo(path, entry, cache_key)
                else:
                    # キャッシュミス: cache_keyをクリアして未キャッシュとしてマーク
                    all_metadata[idx].cache_key = None
                    uncached_metadata.append(all_metadata[idx])

        # キャッシュヒットがない場合は早期リターン
        cached_indices = [
            i for i, entry in enumerate(cached_entries) if entry is not None
        ]
        if not cached_indices:
            return [None] * len(paths), all_metadata

        # 第2パス: ImageMetricsを構築（キャッシュヒット時）
        cached_results: List[Optional[ImageMetrics]] = [None] * len(paths)
        for i, idx in enumerate(cached_indices):
            entry_info = cast(CacheEntryInfo, cached_entries[idx])
            semantic = entry_info.entry.semantic_score
            # CLIP特徴とHSV特徴を結合
            combined_features = np.concatenate(
                [entry_info.entry.hsv_features, entry_info.entry.clip_features]
            )
            # キャッシュに保存されている値を使用
            norm = entry_info.entry.normalized_metrics
            total = entry_info.entry.total_score
            cached_results[idx] = ImageMetrics(
                entry_info.path,
                entry_info.entry.raw_metrics,
                norm,
                semantic,
                total,
                combined_features,
            )

        return cached_results, all_metadata

    def process_batch(
        self,
        paths: List[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> List[Optional[ImageMetrics]]:
        """複数の画像をバッチ処理で解析する.

        2段パイプライン構成:
        1. I/O+CV前処理ステージを並列（ThreadPoolExecutor）
        2. CLIPステージはバッチで直列

        メモリ効率化のためチャンク単位でストリーミング処理:
        - メモリ予算に基づいて動的にチャンクサイズを決定
        - チャンク単位で「前処理→CLIP→結果確定→解放」を実行
        - 大規模画像時のスワップ回避で速度安定化

        キャッシュ機能:
        - まずキャッシュから結果を取得
        - 未キャッシュの画像のみバッチ処理を実行
        - 処理結果はキャッシュに保存

        パフォーマンス最適化:
        - resolve/statの結果をPathMetadataとして保持し重複を回避
        - GPU→CPU転送をチャンク単位でまとめて実行

        Args:
            paths: 画像ファイルパスのリスト
            batch_size: CLIP推論のバッチサイズ（デフォルト32）
            show_progress: 進捗表示をするかどうか

        Returns:
            解析結果のリスト（失敗した画像はNone）
        """
        # ステージ0: キャッシュから結果を取得（メタ情報も収集）
        cached_results, all_metadata = self._get_cached_results(paths)

        # 全てキャッシュにヒットした場合はそのまま返す
        # メタ情報のcache_keyがNoneのものを未キャッシュとして判定
        uncached_metadata = [m for m in all_metadata if m.cache_key is None]
        if not uncached_metadata:
            return cached_results

        # 未キャッシュのパスのみ抽出
        uncached_paths = [m.path for m in uncached_metadata]

        results: List[Optional[ImageMetrics]] = [None] * len(paths)
        # キャッシュ済みの結果を先に埋めておく
        for i, cached_result in enumerate(cached_results):
            if cached_result is not None:
                results[i] = cached_result

        # メモリ予算に基づいてチャンク境界を計算（未キャッシュ分のみ）
        chunk_boundaries = self._compute_chunk_boundaries(
            uncached_paths,
            self.config.max_memory_mb,
            self.config.min_chunk_size,
        )

        # 未キャッシュのメタ情報を元のインデックスにマッピング
        uncached_indices = [
            i for i, m in enumerate(all_metadata) if m.cache_key is None
        ]

        for chunk_idx, (chunk_start, chunk_end) in enumerate(chunk_boundaries):
            # uncached_metadata内のチャンク
            chunk_metadata = uncached_metadata[chunk_start:chunk_end]
            chunk_paths = [m.path for m in chunk_metadata]

            # ステージ1: チャンク単位でI/O + 前処理を並列実行
            pil_images = BatchPipeline.load_and_preprocess_images(chunk_paths)

            # ステージ2: チャンク単位でバッチCLIP推論を実行
            clip_features_list = self.feature_extractor.extract_clip_features_batch(
                pil_images, initial_batch_size=batch_size
            )

            # ステージ3: セマンティックスコアをバッチ計算
            semantic_scores = self.metric_calculator.calculate_semantic_score_batch(
                clip_features_list
            )

            # ステージ3.5: チャンク単位でまとめてGPU→CPU転送
            clip_features_np_list = self._batch_convert_clip_features_to_numpy(
                clip_features_list
            )

            # ステージ4: チャンク単位で結果を構築（並列化）
            # 結果のインデックスを正しくマッピングするためのオフセット
            result_offset = uncached_indices[chunk_start]
            chunk_results = self._process_result_parallel(
                chunk_paths=chunk_paths,
                pil_images=pil_images,
                clip_features_list=clip_features_np_list,
                semantic_scores=semantic_scores,
                chunk_metadata=chunk_metadata,
                chunk_start=result_offset,
                total_paths=len(paths),
                show_progress=show_progress,
            )

            # 結果を正しい位置にマッピング
            for i, chunk_result in enumerate(chunk_results):
                original_idx = uncached_indices[chunk_start + i]
                results[original_idx] = chunk_result

            # チャンク処理完了後にメモリを解放
            del pil_images, clip_features_list, clip_features_np_list

        return results

    @staticmethod
    def _compute_chunk_boundaries(
        paths: List[str], max_memory_mb: int, min_chunk_size: int
    ) -> List[tuple[int, int]]:
        """メモリ予算に基づいてチャンク境界を計算する.

        各画像のファイルサイズを取得し、指定されたメモリ予算を超えない
        ように動的にチャンクを分割する。

        Args:
            paths: 画像ファイルパスのリスト
            max_memory_mb: チャンクあたりの最大メモリ予算（MB）
            min_chunk_size: 最低限確保するチャンクサイズ

        Returns:
            (start_index, end_index) のタプルリスト
        """
        max_memory_bytes = max_memory_mb * 1024 * 1024
        chunks = []
        current_start = 0
        current_memory = 0
        current_count = 0

        for i, path in enumerate(paths):
            # os.statでファイルサイズを取得し、メモリを見積もる
            try:
                file_size = os.path.getsize(path)
                estimated_memory = int(
                    file_size * BatchPipeline.MEMORY_ESTIMATION_FACTOR
                )
            except (OSError, ValueError):
                estimated_memory = 0

            # チャンク追加でメモリ予算を超える場合は新規チャンクを検討
            would_exceed = current_memory + estimated_memory > max_memory_bytes
            has_min_images = current_count >= min_chunk_size
            can_split = i > 0  # 最初の要素で分割しない

            if would_exceed and has_min_images and can_split:
                # 現在のチャンクを確定
                chunks.append((current_start, i))
                current_start = i
                current_memory = estimated_memory
                current_count = 1
            else:
                # 現在のチャンクに追加
                current_memory += estimated_memory
                current_count += 1

        # 最後のチャンクを追加
        if current_start < len(paths):
            final_chunk = (current_start, len(paths))
            # 最後のチャンクがmin_chunk_size未満で、前のチャンクがある場合はマージ
            if chunks and final_chunk[1] - final_chunk[0] < min_chunk_size:
                # 前のチャンクとマージ
                prev_start, _ = chunks.pop()
                chunks.append((prev_start, final_chunk[1]))
            else:
                chunks.append(final_chunk)

        return chunks

    @staticmethod
    def load_and_preprocess_images(
        paths: List[str], max_workers: Optional[int] = None
    ) -> List[Optional[Image.Image]]:
        """複数のパスからPIL画像を読み込み、前処理まで並列実行する.

        I/O（画像読み込み）とCPU-bound処理（RGB変換）をThreadPoolExecutorで並列化.

        Args:
            paths: 画像ファイルパスのリスト
            max_workers: スレッドプールの最大ワーカー数（Noneで自動設定）

        Returns:
            PIL画像のリスト（失敗したパスはNone）
        """

        def process_single(path: str) -> Optional[Image.Image]:
            """単一の画像を読み込み、前処理する."""
            try:
                # PILで画像を読み込み
                with Image.open(path) as img_file:
                    # RGBモードに変換（必要な場合）
                    if img_file.mode != "RGB":
                        pil_img: Image.Image = img_file.convert("RGB")
                        return pil_img.copy()
                    else:
                        return img_file.copy()

            except ExceptionHandler.get_expected_image_errors():
                return None

        # ThreadPoolExecutorで並列処理
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_single, paths))

        return results

    def _process_single_result(
        self,
        path: str,
        pil_img: Image.Image,
        clip_features: np.ndarray,
        semantic: float,
        metadata: PathMetadata | None = None,
    ) -> tuple[Optional[ImageMetrics], dict[str, Any] | None]:
        """結果構築の単一画像処理.

        raw metric計算、feature結合、総合スコア計算を行う。
        キャッシュが有効な場合は、キャッシュエントリを返す。

        Args:
            path: 画像ファイルパス
            pil_img: PIL画像
            clip_features: CLIP特徴（np.ndarray、CPU上）
            semantic: セマンティックスコア
            metadata: パスメタ情報（指定時はresolve/statを再利用）

        Returns:
            (ImageMetricsオブジェクト, キャッシュエントリ) のタプル
            処理失敗時は (None, None)
        """
        try:
            # メトリクス計算用に先に画像を縮小
            # （長辺max_dim px、アスペクト比保持）
            # フル解像度のままnp.array変換→後で縮小の二重処理を回避
            w, h = pil_img.size
            max_dim = self.config.max_dim
            if max(w, h) > max_dim:
                scale = max_dim / max(w, h)
                new_w, new_h = int(w * scale), int(h * scale)
                # PILのthumbnailを使用してアスペクト比を保持しつつ縮小
                pil_img_resized = pil_img.copy()
                pil_img_resized.thumbnail((new_w, new_h), Image.Resampling.BILINEAR)
                img = ImageUtils.pil_to_cv2(pil_img_resized)
            else:
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

            # キャッシュエントリを構築（キャッシュが有効な場合）
            cache_entry: dict[str, Any] | None = None
            if self.cache:
                try:
                    # メタ情報が渡されている場合は再利用
                    if metadata and metadata.cache_key is not None:
                        cache_key = metadata.cache_key
                        absolute_path = metadata.absolute_path or str(
                            Path(path).resolve()
                        )
                        file_stat = metadata.file_stat or Path(path).stat()
                    else:
                        # 従来通り取得（フォールバック）
                        absolute_path = os.path.abspath(path)
                        file_stat = Path(path).stat()
                        cache_key = self.cache.generate_cache_key(
                            absolute_path=absolute_path,
                            file_size=file_stat.st_size,
                            mtime_ns=int(file_stat.st_mtime_ns),
                            model_name=self.model_name,
                            target_text=self.target_text,
                            max_dim=self.config.max_dim,
                        )
                    # HSV特徴は結合ベクトルの前半64要素
                    hsv_features = features[:64]

                    cache_entry = {
                        "cache_key": cache_key,
                        "clip_features": clip_features,
                        "raw_metrics": raw,
                        "hsv_features": hsv_features,
                        "semantic_score": semantic,
                        "normalized_metrics": norm,
                        "total_score": total,
                    }
                except Exception as e:
                    # キャッシュエントリ構築に失敗しても処理は継続
                    logger.debug(
                        f"キャッシュエントリ構築に失敗しました: {path}, 理由: {e}"
                    )

            return ImageMetrics(path, raw, norm, semantic, total, features), cache_entry
        except ExceptionHandler.get_expected_image_errors() as e:
            logger.warning(
                f"画像分析をスキップしました: {path}, 理由: {type(e).__name__}: {e}"
            )
            return None, None

    def _process_result_parallel(
        self,
        chunk_paths: List[str],
        pil_images: List[Optional[Image.Image]],
        clip_features_list: List[Optional[np.ndarray]],
        semantic_scores: List[Optional[float]],
        chunk_metadata: List[PathMetadata],
        chunk_start: int,
        total_paths: int,
        show_progress: bool,
    ) -> List[Optional[ImageMetrics]]:
        """結果構築（raw metric + feature結合）を並列処理.

        ThreadPoolExecutorを使用してチャンク内の画像処理を並列化する。

        Args:
            chunk_paths: チャンク内の画像パスリスト
            pil_images: PIL画像リスト
            clip_features_list: CLIP特徴リスト（np.ndarray、CPU上）
            semantic_scores: セマンティックスコアリスト
            chunk_metadata: チャンク内のパスメタ情報リスト
            chunk_start: チャンクの開始インデックス
            total_paths: 総画像数
            show_progress: 進捗表示フラグ

        Returns:
            ImageMetricsオブジェクトのリスト（失敗時はNone）
        """
        # 並列処理するタスクを準備
        tasks: list[
            tuple[
                int,
                tuple[str, Image.Image, np.ndarray, float, int, PathMetadata] | None,
            ]
        ] = []
        for i, (path, pil_img, clip_features, semantic, metadata) in enumerate(
            zip(
                chunk_paths,
                pil_images,
                clip_features_list,
                semantic_scores,
                chunk_metadata,
            )
        ):
            if pil_img is None or clip_features is None or semantic is None:
                tasks.append((i, None))  # Noneは処理スキップを示す
            else:
                # タスクを保存（インデックス付きで順序維持）
                tasks.append(
                    (
                        i,
                        (
                            path,
                            pil_img,
                            clip_features,
                            semantic,
                            chunk_start + i,
                            metadata,
                        ),
                    )
                )

        # 並列処理関数
        def process_task(
            task_info: tuple[
                int,
                tuple[str, Image.Image, np.ndarray, float, int, PathMetadata] | None,
            ],
        ) -> tuple[int, Optional[ImageMetrics], dict[str, Any] | None]:
            """単一タスクを処理する."""
            idx, data = task_info
            if data is None:
                return idx, None, None
            path, pil_img, clip_features, semantic, global_idx, metadata = data
            if show_progress and global_idx % 50 == 0:
                print(f"解析済み: {global_idx}/{total_paths}")
            result, cache_entry = self._process_single_result(
                path, pil_img, clip_features, semantic, metadata
            )
            return idx, result, cache_entry

        # ThreadPoolExecutorで並列処理
        with ThreadPoolExecutor(max_workers=self._result_max_workers) as executor:
            executor_results = list(executor.map(process_task, tasks))

        # 結果とキャッシュエントリを分離
        results: list[Optional[ImageMetrics]] = [None] * len(tasks)
        cache_entries: list[dict[str, Any]] = []
        for idx, result, cache_entry in executor_results:
            results[idx] = result
            if cache_entry is not None:
                cache_entries.append(cache_entry)

        # メインスレッドでキャッシュを一括保存
        if self.cache and cache_entries:
            try:
                self.cache.put_batch(cache_entries)
            except Exception as e:
                logger.debug(f"キャッシュ一括保存に失敗しました: {e}")

        return results
