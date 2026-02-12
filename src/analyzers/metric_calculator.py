"""メトリクス計算器 - 生メトリクス、セマンティックスコア、総合スコアの計算を行う."""

import logging

import cv2
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

from ..models.analyzer_config import AnalyzerConfig
from .clip_model_manager import CLIPModelManager
from .metric_normalizer import MetricNormalizer

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

        注: 呼出し元（batch_pipeline.py）で既にmax_dimまで縮小された画像を
        受け取ることを想定している。高解像度画像での二重リサイズを回避し、
        メモリと計算コストを削減する。

        Args:
            img: OpenCV画像（BGR形式、既にmax_dim以下に縮小されている）

        Returns:
            生メトリクスの辞書
        """
        # 念のため、画像サイズがmax_dimを超えている場合のみ縮小
        # （通常はbatch_pipeline側で既に縮小済みのためスキップされる）
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

        # OpenCVネイティブ関数で高速化（CV_32Fで精度は維持しつつ高速化）
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        blur_score = float(laplacian.var())

        # 標準偏差はcv2.meanStdDevで計算（高速化）
        _, contrast_std = cv2.meanStdDev(gray)
        contrast = float(contrast_std[0][0])

        # エッジ密度：cv2.countNonZeroで高速化
        edges = cv2.Canny(gray, 50, 150)
        edge_density = cv2.countNonZero(edges) / gray_size

        # UI密度：SobelをCV_32Fで計算し、二乗和の平方根でL1ノルム相当を高速計算
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        ui_density = float(np.sum(np.abs(sobel_x))) / gray_size

        # 彩度の標準偏差をmeanStdDevで高速化
        _, saturation_std = cv2.meanStdDev(hsv[:, :, 1])
        color_richness = float(saturation_std[0][0])

        # アクション強度：フィルタ適用後の標準偏差をmeanStdDevで高速化
        filtered = cv2.filter2D(gray, -1, kernel)
        _, action_std = cv2.meanStdDev(filtered)
        action_intensity = float(action_std[0][0])

        # ドラマティックスコア：彩度と明度の閾値処理にcountNonZeroを活用
        high_saturation = hsv[:, :, 1] > 180
        high_value = hsv[:, :, 2] > 180
        # 両条件を満たすピクセル数をカウント（論理積をuint8に変換してcountNonZero）
        dramatic_pixels = cv2.countNonZero(
            (high_saturation & high_value).astype(np.uint8)
        )
        dramatic_score = (dramatic_pixels / gray_size) * 1000

        return {
            "blur_score": blur_score,
            "brightness": float(gray_mean),
            "contrast": contrast,
            "edge_density": edge_density,
            "color_richness": color_richness,
            "ui_density": ui_density,
            "action_intensity": action_intensity,
            "visual_balance": float(max(0, 100 - abs(gray_mean - 128) * 0.5)),
            "dramatic_score": dramatic_score,
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

    def calculate_semantic_score_batch(
        self, clip_features_list: list[torch.Tensor | None]
    ) -> list[float | None]:
        """複数のCLIP特徴からセマンティックスコアをバッチ計算する.

        パフォーマンス最適化:
        - torch.Tensorを直接受け取り、CPU/NumPy変換を回避
        - まとめて行列積で一括計算

        Args:
            clip_features_list: 正規化済みのCLIP画像特徴のリスト
                                （512次元、torch.Tensor、Noneを含む場合あり）

        Returns:
            セマンティックスコアのリスト（範囲: [-1, 1]、失敗した要素はNone）
        """
        # 有効な特徴のインデックスと特徴を収集
        valid_indices = [
            i for i, features in enumerate(clip_features_list) if features is not None
        ]

        if not valid_indices:
            return [None] * len(clip_features_list)

        # 結果を格納する配列（初期値はNone）
        results: list[float | None] = [None] * len(clip_features_list)

        with torch.inference_mode():
            # torch.Tensorをスタックしてバッチ化
            # valid_indicesでNoneを除外済みだが、型チェッカーに明示するためにcast
            from typing import cast

            valid_tensors: list[torch.Tensor] = [
                cast(torch.Tensor, clip_features_list[i]) for i in valid_indices
            ]
            batch_features = torch.stack(valid_tensors)

            # キャッシュされたテキスト埋め込み（既にL2正規化済み）との
            # コサイン類似度を一括計算
            text_embeddings = self.model_manager.get_text_embeddings()
            cosine_sims = torch.matmul(batch_features, text_embeddings.T)

            # 結果を元のインデックスにマッピング
            for j, idx in enumerate(valid_indices):
                results[idx] = float(cosine_sims[j][0])

        return results

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

    def calculate_raw_norm_metrics(
        self, img: np.ndarray
    ) -> tuple[dict[str, float], dict[str, float]]:
        """生メトリクスと正規化メトリクスのみ計算する.

        セマンティックスコアはバッチ計算済みの値を使用するため、
        このメソッドでは計算しない。

        Args:
            img: OpenCV画像（BGR形式）

        Returns:
            (生メトリクス, 正規化メトリクス)のタプル
        """
        raw = self.calculate_raw_metrics(img)
        norm = MetricNormalizer.normalize_all(raw)
        return raw, norm

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
