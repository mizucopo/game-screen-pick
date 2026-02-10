"""Image quality analyzer using CLIP and computer vision metrics."""

import logging
from typing import Optional
import numpy as np
import cv2
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from PIL import UnidentifiedImageError

from ..models.image_metrics import ImageMetrics
from ..models.genre_weights import GenreWeights
from .metric_normalizer import MetricNormalizer

logger = logging.getLogger(__name__)


class ImageQualityAnalyzer:
    """画像品質アナライザー."""

    def __init__(self, genre: str = "mixed"):
        """アナライザーを初期化する."""
        self.weights = GenreWeights.get_weights(genre)
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def _extract_diversity_features(self, img: np.ndarray) -> np.ndarray:
        """見た目の特徴を抽出（色と構造）."""
        small = cv2.resize(img, (128, 128))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
        return cv2.normalize(hist, hist).flatten()

    def analyze(self, path: str) -> Optional[ImageMetrics]:
        """画像を解析して品質スコアを計算する."""
        # 正常な失敗（画像ファイルの問題、OpenCV処理で破損画像を検出、配列操作不可）
        expected_errors = (
            FileNotFoundError,
            UnidentifiedImageError,
            OSError,
            cv2.error,
            ValueError,
        )
        # 異常な失敗（実装バグ）
        unexpected_errors = (
            AttributeError,
            TypeError,
            KeyError,
            IndexError,
            RuntimeError,
            torch.cuda.OutOfMemoryError,
            MemoryError,
        )

        try:
            img = cv2.imread(path)
            if img is None:
                return None
            features = self._extract_diversity_features(img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            gray_size = gray.size
            img_size = img.size
            gray_mean = np.mean(gray)

            kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            raw = {
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
                    np.sum((hsv[:, :, 1] > 180) & (hsv[:, :, 2] > 180)) / img_size
                )
                * 1000,
            }
            norm = MetricNormalizer.normalize_all(raw)
            with torch.no_grad():
                with Image.open(path) as pil_img:
                    inputs = self.processor(
                        text=["epic game scenery"],
                        images=pil_img,
                        return_tensors="pt",
                        padding=True,
                    ).to(self.device)
                    semantic = (
                        float(self.model(**inputs).logits_per_image[0][0]) / 100.0
                    )

            weighted_sum = sum(
                norm[k] * self.weights.get(k, 0.0) for k in norm if k in self.weights
            )
            # 暗すぎる画像にはペナルティを適用
            penalty = 0.6 if raw["brightness"] < 40 else 0.0
            total = max(0.0, (weighted_sum + (semantic * 0.2) - penalty) * 100.0)
            return ImageMetrics(path, raw, norm, semantic, total, features)
        except expected_errors as e:
            logger.warning(
                f"画像分析をスキップしました: {path}, 理由: {type(e).__name__}: {e}"
            )
            return None
        except unexpected_errors:
            logger.error(f"予期しないエラーが発生しました: {path}", exc_info=True)
            raise
