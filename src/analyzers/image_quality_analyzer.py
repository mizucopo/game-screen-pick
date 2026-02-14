"""Image quality analyzer using CLIP and computer vision metrics."""

import logging

from ..constants.score_weights import ScoreWeights
from ..models.analyzer_config import AnalyzerConfig
from ..models.image_metrics import ImageMetrics
from ..utils.image_utils import ImageUtils
from .batch_pipeline import BatchPipeline
from .clip_model_manager import CLIPModelManager
from .feature_extractor import FeatureExtractor
from .metric_calculator import MetricCalculator

logger = logging.getLogger(__name__)


class ImageQualityAnalyzer:
    """画像品質アナライザー（Facadeパターン）.

    複数のコンポーネントを統合し、公開APIを提供する。
    CLIPモデルでセマンティック特徴を抽出し、画像メトリクスを計算する。
    """

    def __init__(
        self,
        config: AnalyzerConfig | None = None,
        device: str | None = None,
    ):
        """アナライザーを初期化する.

        Args:
            config: アナライザー設定（Noneの場合はデフォルト値を使用）
            device: 使用するデバイス（Noneの場合は自動検出）
        """
        self.config = config or AnalyzerConfig()
        self.weights = ScoreWeights.get_weights()

        # レイヤー1: モデル管理（プライベート）
        self._model_manager = CLIPModelManager(
            target_text="epic game scenery", device=device
        )

        # レイヤー2: 特徴抽出（モデルに依存）
        self.feature_extractor = FeatureExtractor(self._model_manager)

        # レイヤー3: スコア計算（モデルのテキスト埋め込みに依存）
        self.metric_calculator = MetricCalculator(
            self.config, self.weights, self._model_manager
        )

        # レイヤー4: バッチ処理（上記全てに依存）
        self.batch_pipeline = BatchPipeline(
            self.feature_extractor,
            self.metric_calculator,
            self.config,
        )

    def analyze(self, path: str) -> ImageMetrics | None:
        """画像を解析して品質スコアを計算する."""
        pil_img_copy = ImageUtils.load_as_rgb(path)
        if pil_img_copy is None:
            logger.warning(f"画像の読み込みに失敗しました: {path}")
            return None

        # OpenCV形式（BGR）に変換
        img = ImageUtils.pil_to_cv2(pil_img_copy)

        # CLIP特徴を抽出
        clip_features = self.feature_extractor.extract_clip_features(pil_img_copy)

        # HSV特徴とCLIP特徴を結合
        features = self.feature_extractor.extract_combined_features(img, clip_features)

        # すべてのメトリクスを一括計算
        raw, norm, semantic, total = self.metric_calculator.calculate_all_metrics(
            img, clip_features
        )

        return ImageMetrics(path, raw, norm, semantic, total, features)

    def analyze_batch(
        self,
        paths: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> list[ImageMetrics | None]:
        """複数の画像をバッチ処理で解析する.

        Args:
            paths: 画像ファイルパスのリスト
            batch_size: CLIP推論のバッチサイズ（デフォルト32）
            show_progress: 進捗表示をするかどうか

        Returns:
            解析結果のリスト（失敗した画像はNone）
        """
        return self.batch_pipeline.process_batch(paths, batch_size, show_progress)
