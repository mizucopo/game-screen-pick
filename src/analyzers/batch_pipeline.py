"""バッチ処理パイプライン - 複数画像のバッチ処理をオーケストレーションする."""

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from ..models.analyzer_config import AnalyzerConfig
from ..models.image_metrics import ImageMetrics

from .feature_extractor import FeatureExtractor
from .metric_calculator import MetricCalculator

logger = logging.getLogger(__name__)


class BatchPipeline:
    """バッチ処理パイプライン.

    複数の画像をバッチ処理で解析するオーケストレーションを行う。
    """

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

        Args:
            paths: 画像ファイルパスのリスト
            batch_size: CLIP推論のバッチサイズ（デフォルト32）
            show_progress: 進捗表示をするかどうか

        Returns:
            解析結果のリスト（失敗した画像はNone）
        """
        results: List[Optional[ImageMetrics]] = []

        # メモリ予算に基づいてチャンク境界を計算
        chunk_boundaries = self._compute_chunk_boundaries(
            paths,
            self.config.max_memory_mb,
            self.config.min_chunk_size,
        )

        for chunk_start, chunk_end in chunk_boundaries:
            chunk_paths = paths[chunk_start:chunk_end]

            # ステージ1: チャンク単位でI/O + 前処理を並列実行
            pil_images = BatchPipeline.load_and_preprocess_images(chunk_paths)

            # ステージ2: チャンク単位でバッチCLIP推論を実行
            clip_features_list = self.feature_extractor.extract_clip_features_batch(
                pil_images, initial_batch_size=batch_size
            )

            # ステージ3: チャンク単位で結果を構築
            for i, (path, pil_img, clip_features) in enumerate(
                zip(chunk_paths, pil_images, clip_features_list)
            ):
                if pil_img is None or clip_features is None:
                    results.append(None)
                    continue

                if show_progress and (chunk_start + i) % 50 == 0:
                    print(f"解析済み: {chunk_start + i}/{len(paths)}")

                try:
                    # OpenCV形式（BGR）に変換
                    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

                    # すべてのメトリクスを一括計算
                    raw, norm, semantic, total = (
                        self.metric_calculator.calculate_all_metrics(
                            img,
                            clip_features,
                        )
                    )

                    # HSV特徴とCLIP特徴を結合
                    features = self.feature_extractor.extract_combined_features(
                        img,
                        clip_features,
                    )
                    results.append(
                        ImageMetrics(path, raw, norm, semantic, total, features)
                    )
                except self._get_expected_errors() as e:
                    logger.warning(
                        f"画像分析をスキップしました: {path}, "
                        f"理由: {type(e).__name__}: {e}"
                    )
                    results.append(None)

            # チャンク処理完了後にメモリを解放
            del pil_images, clip_features_list

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
            # ファイルサイズを取得（存在しない場合は0とする）
            try:
                file_size = os.path.getsize(path)
                # 推定メモリ使用量はファイルサイズの約2倍（デコード後）とする
                estimated_memory = file_size * 2
            except OSError:
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

            except (FileNotFoundError, UnidentifiedImageError, OSError, ValueError):
                return None

        # ThreadPoolExecutorで並列処理
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_single, paths))

        return results

    @staticmethod
    def _get_expected_errors() -> tuple[type[Exception], ...]:
        """正常な失敗として扱うエラー型."""
        return (
            FileNotFoundError,
            UnidentifiedImageError,
            OSError,
            cv2.error,
            ValueError,
        )
