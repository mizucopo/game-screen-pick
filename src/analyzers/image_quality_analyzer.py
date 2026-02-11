"""Image quality analyzer using CLIP and computer vision metrics."""

import logging
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from ..models.analyzer_config import AnalyzerConfig
from ..models.image_metrics import ImageMetrics
from ..models.genre_weights import GenreWeights
from .batch_pipeline import BatchPipeline
from .clip_model_manager import CLIPModelManager
from .feature_extractor import FeatureExtractor
from .metric_calculator import MetricCalculator

logger = logging.getLogger(__name__)


def _safe_l2_normalize(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """ゼロ割れ安全なL2正規化を行う.

    Args:
        vec: 正規化するベクトル
        eps: ゼロ割れ防止用の微小値

    Returns:
        L2正規化されたベクトル（元のノルムが0の場合はゼロベクトル）
    """
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return np.zeros_like(vec)
    return vec / norm


class ImageQualityAnalyzer:
    """画像品質アナライザー（Facadeパターン）.

    複数のコンポーネントを統合し、公開APIを提供する。
    """

    def __init__(self, genre: str = "mixed", config: AnalyzerConfig | None = None):
        """アナライザーを初期化する.

        Args:
            genre: ジャンル（重み付け用）
            config: アナライザー設定（Noneの場合はデフォルト値を使用）
        """
        self.config = config or AnalyzerConfig()
        self.weights = GenreWeights.get_weights(genre)

        # レイヤー1: モデル管理（プライベート）
        self._model_manager = CLIPModelManager(target_text="epic game scenery")

        # レイヤー2: 特徴抽出（モデルに依存）
        self.feature_extractor = FeatureExtractor(self._model_manager)

        # レイヤー3: スコア計算（モデルのテキスト埋め込みに依存）
        self.metric_calculator = MetricCalculator(
            self.config, self.weights, self._model_manager
        )

        # レイヤー4: バッチ処理（上記全てに依存）
        self.batch_pipeline = BatchPipeline(
            self.feature_extractor, self.metric_calculator, self.config
        )

    def analyze(self, path: str) -> Optional[ImageMetrics]:
        """画像を解析して品質スコアを計算する."""
        try:
            # PILで1回だけ読み込み、ファイル記述子のリークを防止
            with Image.open(path) as pil_img:
                # RGBモードに変換（必要な場合）
                if pil_img.mode != "RGB":
                    pil_img_rgb: Image.Image = pil_img.convert("RGB")
                    pil_img_copy = pil_img_rgb.copy()
                else:
                    pil_img_copy = pil_img.copy()

                # OpenCV形式（BGR）に変換
                img = cv2.cvtColor(np.array(pil_img_copy), cv2.COLOR_RGB2BGR)

                # CLIP特徴を抽出
                clip_features = self.feature_extractor.extract_clip_features(
                    pil_img_copy
                )

                # HSV特徴とCLIP特徴を結合
                features = self.feature_extractor.extract_combined_features(
                    img, clip_features
                )

                # すべてのメトリクスを一括計算
                raw, norm, semantic, total = (
                    self.metric_calculator.calculate_all_metrics(img, clip_features)
                )

                return ImageMetrics(path, raw, norm, semantic, total, features)
        except self._get_expected_errors() as e:
            logger.warning(
                f"画像分析をスキップしました: {path}, 理由: {type(e).__name__}: {e}"
            )
            return None
        except self._get_unexpected_errors():
            logger.error(f"予期しないエラーが発生しました: {path}", exc_info=True)
            raise

    def analyze_batch(
        self,
        paths: List[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> List[Optional[ImageMetrics]]:
        """複数の画像をバッチ処理で解析する.

        Args:
            paths: 画像ファイルパスのリスト
            batch_size: CLIP推論のバッチサイズ（デフォルト32）
            show_progress: 進捗表示をするかどうか

        Returns:
            解析結果のリスト（失敗した画像はNone）
        """
        return self.batch_pipeline.process_batch(paths, batch_size, show_progress)

    # 後方互換性のためのプロパティとメソッド
    @property
    def model(self):  # type: ignore[no-untyped-def]
        """モデルにアクセスするための後方互換性プロパティ."""
        return self._model_manager.model

    @property
    def processor(self):  # type: ignore[no-untyped-def]
        """プロセッサにアクセスするための後方互換性プロパティ."""
        return self._model_manager.processor

    @property
    def device(self):  # type: ignore[no-untyped-def]
        """デバイスにアクセスするための後方互換性プロパティ."""
        return self._model_manager.device

    def _extract_hsv_features(self, img: np.ndarray) -> np.ndarray:
        """HSV特徴を抽出する（後方互換性）."""
        return self.feature_extractor.extract_hsv_features(img)

    def _extract_clip_features(self, pil_img: Image.Image) -> np.ndarray:
        """CLIP特徴を抽出する（後方互換性）."""
        return self.feature_extractor.extract_clip_features(pil_img)

    def _extract_combined_features(
        self, img: np.ndarray, clip_features: np.ndarray
    ) -> np.ndarray:
        """統合特徴を抽出する（後方互換性）."""
        return self.feature_extractor.extract_combined_features(img, clip_features)

    def _calculate_raw_metrics(self, img: np.ndarray) -> dict[str, float]:
        """生メトリクスを計算する（後方互換性）."""
        return self.metric_calculator.calculate_raw_metrics(img)

    def _calculate_semantic_score(self, pil_img: Image.Image) -> float:
        """セマンティックスコアを計算する（後方互換性）."""
        return self.metric_calculator.calculate_semantic_score(pil_img)

    def _calculate_semantic_score_from_features(
        self, clip_features: np.ndarray
    ) -> float:
        """特徴からセマンティックスコアを計算する（後方互換性）."""
        return self.metric_calculator.calculate_semantic_score_from_features(
            clip_features
        )

    def _calculate_total_score(
        self, raw: dict[str, float], norm: dict[str, float], semantic: float
    ) -> float:
        """総合スコアを計算する（後方互換性）."""
        return self.metric_calculator.calculate_total_score(raw, norm, semantic)

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

    @staticmethod
    def _get_unexpected_errors() -> tuple[type[Exception], ...]:
        """異常な失敗として扱うエラー型（実装バグ）."""
        return (
            AttributeError,
            TypeError,
            KeyError,
            IndexError,
            RuntimeError,
            MemoryError,
        )
