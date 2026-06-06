"""BatchPipelineテスト用feature extractor."""

import numpy as np
import torch
from PIL import Image


class FakeFeatureExtractor:
    """BatchPipeline向けの軽量特徴抽出器."""

    @staticmethod
    def _hsv_features_or_default(hsv_features: np.ndarray | None) -> np.ndarray:
        if hsv_features is not None:
            return hsv_features
        return np.ones(64, dtype=np.float32)

    @staticmethod
    def extract_clip_features_batch(
        pil_images: list[Image.Image | None],
        initial_batch_size: int = 32,
    ) -> list[torch.Tensor | None]:
        del initial_batch_size
        return [
            torch.ones(512, dtype=torch.float32) if image is not None else None
            for image in pil_images
        ]

    @staticmethod
    def extract_hsv_features(img: np.ndarray) -> np.ndarray:
        del img
        return np.ones(64, dtype=np.float32)

    @staticmethod
    def extract_combined_features(
        img: np.ndarray,
        clip_features: np.ndarray,
        hsv_features: np.ndarray | None = None,
    ) -> np.ndarray:
        del img
        features = FakeFeatureExtractor._hsv_features_or_default(hsv_features)
        return np.concatenate([features, clip_features])

    @staticmethod
    def extract_content_features(
        img: np.ndarray,
        raw_metrics: object,
        hsv_features: np.ndarray | None = None,
    ) -> np.ndarray:
        del img, raw_metrics
        features = FakeFeatureExtractor._hsv_features_or_default(hsv_features)
        return np.concatenate([features, np.ones(37, dtype=np.float32)])
