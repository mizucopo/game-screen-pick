"""Image quality analyzer using CLIP and computer vision metrics."""

from typing import Optional
import numpy as np
import cv2
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

from ..models.image_metrics import ImageMetrics
from ..models.genre_weights import GenreWeights
from .metric_normalizer import MetricNormalizer


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
        try:
            img = cv2.imread(path)
            if img is None:
                return None
            features = self._extract_diversity_features(img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            raw = {
                "blur_score": cv2.Laplacian(gray, cv2.CV_64F).var(),
                "brightness": np.mean(gray),
                "contrast": np.std(gray),
                "edge_density": np.sum(cv2.Canny(gray, 50, 150) > 0) / gray.size,
                "color_richness": np.std(hsv[:, :, 1]),
                "ui_density": (
                    np.sum(np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0))) /
                    gray.size
                ),
                "action_intensity": np.std(cv2.filter2D(
                    gray, -1, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
                )),
                "visual_balance": max(0, 100 - abs(np.mean(gray) - 128) * 0.5),
                "dramatic_score": (
                    np.sum((hsv[:, :, 1] > 180) & (hsv[:, :, 2] > 180)) /
                    img.size
                ) * 1000
            }
            norm = MetricNormalizer.normalize_all(raw)
            with torch.no_grad():
                inputs = self.processor(
                    text=["epic game scenery"],
                    images=Image.open(path),
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                semantic = float(self.model(**inputs).logits_per_image[0][0]) / 100.0

            weighted_sum = sum(
                norm[k] * self.weights.get(k, 0.0)
                for k in norm
                if k in self.weights
            )
            # ペナルティ（暗すぎる画像）
            penalty = 0.6 if raw['brightness'] < 40 else 0.0
            total = max(0.0, (weighted_sum + (semantic * 0.2) - penalty) * 100.0)
            return ImageMetrics(path, raw, norm, semantic, total, features)
        except:
            return None
