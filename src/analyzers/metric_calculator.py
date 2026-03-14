"""メトリクス計算器."""

import cv2
import numpy as np

from ..analyzers.metric_normalizer import MetricNormalizer
from ..models.analyzer_config import AnalyzerConfig
from ..models.normalized_metrics import NormalizedMetrics
from ..models.raw_metrics import RawMetrics


class MetricCalculator:
    """画像メトリクス計算器.

    生メトリクス、正規化メトリクス、品質スコア補助値を計算する。
    """

    def __init__(
        self,
        config: AnalyzerConfig,
    ):
        """メトリクス計算器を初期化する.

        Args:
            config: アナライザー設定
        """
        self.config = config

    def calculate_raw_metrics(self, img: np.ndarray) -> RawMetrics:
        """生の画像メトリクスを計算する.

        注: 呼び出し元（batch_pipeline.py）で既にmax_dimまで縮小された画像を
        受け取ることを想定している。高解像度画像での二重リサイズを回避し、
        メモリと計算コストを削減する。

        Args:
            img: OpenCV画像（BGR形式、既にmax_dim以下に縮小されている）

        Returns:
            RawMetricsインスタンス
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
        gray_mean = cv2.mean(gray)[0]
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

        # UI密度：SobelをCV_32Fで計算し、cv2.normでL1ノルムを高速計算
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        ui_density = cv2.norm(sobel_x, cv2.NORM_L1) / gray_size

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
        visual_balance = float(max(0, 100 - abs(gray_mean - 128) * 0.5))

        return RawMetrics(
            blur_score=blur_score,
            brightness=float(gray_mean),
            contrast=contrast,
            edge_density=edge_density,
            color_richness=color_richness,
            ui_density=ui_density,
            action_intensity=action_intensity,
            visual_balance=visual_balance,
            dramatic_score=dramatic_score,
        )

    def calculate_quality_score(
        self, norm: NormalizedMetrics, weights: dict[str, float]
    ) -> float:
        """品質スコアを計算する."""
        metric_names = (
            "blur_score",
            "contrast",
            "color_richness",
            "edge_density",
            "dramatic_score",
            "visual_balance",
            "action_intensity",
            "ui_density",
        )
        return float(
            sum(
                getattr(norm, metric_name) * weights.get(metric_name, 0.0)
                for metric_name in metric_names
            )
        )

    def calculate_brightness_penalty(self, raw: RawMetrics) -> float:
        """暗すぎる画像へのペナルティを返す."""
        if raw.brightness < self.config.brightness_penalty_threshold:
            return self.config.brightness_penalty_value
        return 0.0

    def calculate_raw_norm_metrics(
        self, img: np.ndarray
    ) -> tuple[RawMetrics, NormalizedMetrics]:
        """生メトリクスと正規化メトリクスのみ計算する.

        Args:
            img: OpenCV画像（BGR形式）

        Returns:
            (RawMetrics, NormalizedMetrics)のタプル
        """
        raw = self.calculate_raw_metrics(img)
        norm = MetricNormalizer.normalize_all(raw)
        return raw, norm

    def calculate_all_metrics(
        self, img: np.ndarray
    ) -> tuple[RawMetrics, NormalizedMetrics]:
        """生メトリクスと正規化メトリクスを一括計算する."""
        raw = self.calculate_raw_metrics(img)
        norm = MetricNormalizer.normalize_all(raw)
        return raw, norm
