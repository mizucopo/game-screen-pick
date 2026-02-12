"""並列メトリクス計算ワーカー.

ProcessPoolExecutorで使用されるワーカークラス。
cv2処理、メトリクス計算、結合特徴抽出を行う。
"""

import logging
from typing import Any, Optional

import cv2
import numpy as np

from ..models.analyzer_config import AnalyzerConfig
from ..models.image_metrics import ImageMetrics
from ..utils.vector_utils import VectorUtils
from .metric_normalizer import MetricNormalizer

logger = logging.getLogger(__name__)


class ParallelMetricsWorker:
    """並列メトリクス計算ワーカークラス.

    cv2処理、メトリクス計算、結合特徴抽出を行う静的メソッドを提供する。
    ProcessPoolExecutorでプロセス間転送されるため、インスタンス化せずstaticメソッドのみを使用する。
    """

    @staticmethod
    def calculate_raw_metrics(img: np.ndarray, max_dim: int) -> dict[str, float]:
        """生の画像メトリクスを計算する.

        Args:
            img: OpenCV画像（BGR形式）
            max_dim: メトリクス計算用の画像リサイズ時の長辺の最大ピクセル数

        Returns:
            生メトリクスの辞書
        """
        # メトリクス計算用に画像を縮小（長辺max_dim px、アスペクト比保持）
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray_size = gray.size
        gray_mean = np.mean(gray)
        kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

        return {
            "blur_score": cv2.Laplacian(gray, cv2.CV_64F).var(),
            "brightness": gray_mean,
            "contrast": np.std(gray),
            "edge_density": np.sum(cv2.Canny(gray, 50, 150) > 0) / gray_size,
            "color_richness": np.std(hsv[:, :, 1]),
            "ui_density": (
                np.sum(np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0))) / gray_size
            ),
            "action_intensity": np.std(cv2.filter2D(gray, -1, kernel)),
            "visual_balance": max(0, 100 - abs(gray_mean - 128) * 0.5),
            "dramatic_score": (
                np.sum((hsv[:, :, 1] > 180) & (hsv[:, :, 2] > 180)) / gray_size
            )
            * 1000,
        }

    @staticmethod
    def extract_hsv_features(img: np.ndarray) -> np.ndarray:
        """HSV色空間のヒストグラム特徴を抽出する.

        Args:
            img: OpenCV画像（BGR形式）

        Returns:
            正規化されたHSVヒストグラム特徴（64次元）
        """
        small = cv2.resize(img, (128, 128))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
        return cv2.normalize(hist, hist).flatten()

    @staticmethod
    def calculate_total_score(
        raw: dict[str, float],
        norm: dict[str, float],
        semantic: float,
        weights: dict[str, float],
        config: AnalyzerConfig,
    ) -> float:
        """総合スコアを計算する.

        Args:
            raw: 生メトリクス
            norm: 正規化されたメトリクス
            semantic: セマンティックスコア
            weights: メトリクスの重み
            config: アナライザー設定

        Returns:
            総合スコア
        """
        weighted_sum = sum(norm[k] * weights.get(k, 0.0) for k in norm if k in weights)
        penalty = (
            config.brightness_penalty_value
            if raw["brightness"] < config.brightness_penalty_threshold
            else 0.0
        )
        return max(
            0.0,
            (weighted_sum + (semantic * config.semantic_weight) - penalty)
            * config.score_multiplier,
        )

    @staticmethod
    def process_single_image(args: tuple[Any, ...]) -> Optional[ImageMetrics]:
        """単一画像のメトリクス計算処理（cv2処理 + メトリクス計算 + 結合特徴抽出）.

        タプル引数を受け取る（ProcessPoolExecutor対応）.

        Args:
            args: (path, pil_img, clip_features, text_embeddings, max_dim,
                   weights, config) のタプル

        Returns:
            ImageMetrics、処理失敗時はNone
        """
        (
            path,
            pil_img,
            clip_features,
            text_embeddings,
            max_dim,
            weights,
            config,
        ) = args

        try:
            # OpenCV形式（BGR）に変換
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            # 生メトリクスを計算
            raw = ParallelMetricsWorker.calculate_raw_metrics(img, max_dim)

            # メトリクスを正規化
            norm = MetricNormalizer.normalize_all(raw)

            # セマンティックスコア計算（コサイン類似度）
            clip_features_normalized = VectorUtils.safe_l2_normalize(clip_features)
            text_embeddings_normalized = VectorUtils.safe_l2_normalize(text_embeddings)
            semantic = float(
                np.dot(clip_features_normalized, text_embeddings_normalized)
            )

            # 総合スコア計算
            total = ParallelMetricsWorker.calculate_total_score(
                raw, norm, semantic, weights, config
            )

            # HSV特徴抽出と結合
            hsv_features = ParallelMetricsWorker.extract_hsv_features(img)
            hsv_normalized = VectorUtils.safe_l2_normalize(hsv_features)
            combined_features = np.concatenate([hsv_normalized, clip_features])

            return ImageMetrics(path, raw, norm, semantic, total, combined_features)

        except (FileNotFoundError, OSError, cv2.error, ValueError):
            return None
