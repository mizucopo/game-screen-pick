"""メトリクス計算器 - 生メトリクス、セマンティックスコア、総合スコアの計算を行う."""

import logging

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ..models.analyzer_config import AnalyzerConfig
from .clip_model_manager import CLIPModelManager
from .metric_normalizer import MetricNormalizer

import PIL.Image

logger = logging.getLogger(__name__)


class MetricCalculator:
    """画像メトリクス計算器.

    生メトリクス、セマンティックスコア、総合スコアを計算する。
    """

    def __init__(
        self,
        config: AnalyzerConfig,
        weights: dict[str, float],
        model_manager: "CLIPModelManager",
    ):
        """メトリクス計算器を初期化する.

        Args:
            config: アナライザー設定
            weights: ジャンル別の重み
            model_manager: CLIPモデルマネージャー
        """
        self.config = config
        self.weights = weights
        self.model_manager = model_manager

    def calculate_raw_metrics(self, img: np.ndarray) -> dict[str, float]:
        """生の画像メトリクスを計算する.

        メトリクス計算用に画像を長辺max_dim pxに縮小して処理することで、
        計算コストを削減する。アスペクト比は保持する。

        Args:
            img: OpenCV画像（BGR形式）

        Returns:
            生メトリクスの辞書
        """
        # メトリクス計算用に画像を縮小（長辺max_dim px、アスペクト比保持）
        h, w = img.shape[:2]
        if max(h, w) > self.config.max_dim:
            scale = self.config.max_dim / max(h, w)
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

    def calculate_semantic_score(self, pil_img: PIL.Image.Image) -> float:
        """CLIPモデルを使用してセマンティックスコアを計算する.

        Args:
            pil_img: PIL画像（RGB形式）

        Returns:
            コサイン類似度ベースのセマンティックスコア（範囲: [-1, 1]）
        """
        with torch.inference_mode():
            inputs = self.model_manager.processor(
                images=pil_img,
                return_tensors="pt",
                padding=True,
            ).to(self.model_manager.device)
            image_features = self.model_manager.model.get_image_features(**inputs)

            # 画像特徴をL2正規化
            image_features_normalized = F.normalize(image_features, p=2, dim=-1)

            # キャッシュされたテキスト埋め込み（既にL2正規化済み）との
            # コサイン類似度を計算
            text_embeddings = self.model_manager.get_text_embeddings()
            cosine_sim = torch.matmul(image_features_normalized, text_embeddings.T)
            return float(cosine_sim[0][0])

    def calculate_semantic_score_from_features(
        self, clip_features: np.ndarray
    ) -> float:
        """既に計算済みのCLIP特徴からセマンティックスコアを計算する.

        Args:
            clip_features: 正規化済みのCLIP画像特徴（512次元）

        Returns:
            コサイン類似度ベースのセマンティックスコア（範囲: [-1, 1]）
        """
        # NumPy配列をtorch.Tensorに変換（float32指定）
        with torch.inference_mode():
            image_features = (
                torch.from_numpy(clip_features.astype(np.float32))
                .unsqueeze(0)
                .to(self.model_manager.device)
            )
            # キャッシュされたテキスト埋め込み（既にL2正規化済み）との
            # コサイン類似度を計算
            text_embeddings = self.model_manager.get_text_embeddings()
            cosine_sim = torch.matmul(image_features, text_embeddings.T)
            return float(cosine_sim[0][0])

    def calculate_total_score(
        self, raw: dict[str, float], norm: dict[str, float], semantic: float
    ) -> float:
        """総合スコアを計算する.

        Args:
            raw: 生メトリクス
            norm: 正規化されたメトリクス
            semantic: セマンティックスコア

        Returns:
            総合スコア
        """
        weighted_sum = sum(
            norm[k] * self.weights.get(k, 0.0) for k in norm if k in self.weights
        )
        penalty = (
            self.config.brightness_penalty_value
            if raw["brightness"] < self.config.brightness_penalty_threshold
            else 0.0
        )
        return max(
            0.0,
            (weighted_sum + (semantic * self.config.semantic_weight) - penalty)
            * self.config.score_multiplier,
        )

    def calculate_all_metrics(
        self, img: np.ndarray, clip_features: np.ndarray
    ) -> tuple[dict[str, float], dict[str, float], float, float]:
        """すべてのメトリクスを一括計算する.

        Args:
            img: OpenCV画像（BGR形式）
            clip_features: CLIP画像特徴（512次元、正規化済み）

        Returns:
            (生メトリクス, 正規化メトリクス, セマンティックスコア, 総合スコア)のタプル
        """
        raw = self.calculate_raw_metrics(img)
        norm = MetricNormalizer.normalize_all(raw)
        semantic = self.calculate_semantic_score_from_features(clip_features)
        total = self.calculate_total_score(raw, norm, semantic)
        return raw, norm, semantic, total
