"""画像解析Facade.

CLIP特徴抽出、画質メトリクス計算、レイアウトヒューリスティクス抽出を
まとめて提供し、scene判定前の中立解析結果を返す。
"""

import logging

from ..models.analyzed_image import AnalyzedImage
from ..models.analyzer_config import AnalyzerConfig
from ..utils.image_utils import ImageUtils
from .batch_pipeline import BatchPipeline
from .clip_model_manager import CLIPModelManager
from .feature_extractor import FeatureExtractor
from .layout_analyzer import LayoutAnalyzer
from .metric_calculator import MetricCalculator

logger = logging.getLogger(__name__)


class ImageQualityAnalyzer:
    """画像品質アナライザー（Facadeパターン）.

    複数のコンポーネントを統合し、公開APIを提供する。
    CLIP特徴と画像メトリクスを抽出し、中立な分析結果を返す。
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

        # レイヤー1: モデル管理（プライベート）
        self._model_manager = CLIPModelManager(device=device)

        # レイヤー2: 特徴抽出（モデルに依存）
        self.feature_extractor = FeatureExtractor(self._model_manager)

        # レイヤー3: メトリクス計算
        self.metric_calculator = MetricCalculator(self.config)

        # レイヤー4: バッチ処理（上記全てに依存）
        self.batch_pipeline = BatchPipeline(
            self.feature_extractor,
            self.metric_calculator,
            self.config,
        )

    def analyze(self, path: str) -> AnalyzedImage | None:
        """画像を解析して中立な特徴を計算する.

        単一画像向けの同期APIで、CLIP特徴、結合特徴、画質メトリクス、
        レイアウトヒューリスティクスを一度に計算する。
        この段階ではscene labelや選定スコアは付与せず、
        後段の `SceneScorer` と `CandidateScorer` が扱う素材だけを返す。

        Args:
            path: 解析対象画像のファイルパス。

        Returns:
            中立解析結果 `AnalyzedImage` 。読み込み失敗時は `None` 。
        """
        pil_img_copy = ImageUtils.load_as_rgb(path)
        if pil_img_copy is None:
            logger.warning(f"画像の読み込みに失敗しました: {path}")
            return None

        # OpenCV形式（BGR）に変換
        img = ImageUtils.pil_to_cv2(pil_img_copy)

        # CLIP特徴を抽出
        clip_features = self.feature_extractor.extract_clip_features(pil_img_copy)

        # すべてのメトリクスを一括計算
        raw, norm = self.metric_calculator.calculate_all_metrics(img)
        combined_features = self.feature_extractor.extract_combined_features(
            img,
            clip_features,
        )
        content_features = self.feature_extractor.extract_content_features(img, raw)
        layout_heuristics = LayoutAnalyzer.analyze(img)

        return AnalyzedImage(
            path=path,
            raw_metrics=raw,
            normalized_metrics=norm,
            clip_features=clip_features,
            combined_features=combined_features,
            content_features=content_features,
            layout_heuristics=layout_heuristics,
        )

    def analyze_batch(
        self,
        paths: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> list[AnalyzedImage | None]:
        """複数の画像をバッチ処理で解析する.

        単一画像APIと同じ中立解析結果を返すが、
        実装は `BatchPipeline` に委譲し、チャンク分割と先読みを用いて
        大量画像をメモリ予算内で処理する。

        Args:
            paths: 画像ファイルパスのリスト
            batch_size: CLIP推論のバッチサイズ（デフォルト32）
            show_progress: 進捗表示をするかどうか

        Returns:
            解析結果のリスト（失敗した画像はNone）
        """
        return self.batch_pipeline.process_batch(paths, batch_size, show_progress)

    def close(self) -> None:
        """保持中のリソースを明示的に解放する.

        内部の BatchPipeline が管理するスレッドプールを
        安全に停止し、再利用可能な状態へ戻す。
        """
        self.batch_pipeline.close()

    def __enter__(self) -> "ImageQualityAnalyzer":
        """コンテキストマネージャーに入る."""
        return self

    def __exit__(self, *args: object) -> None:
        """コンテキストマネージャーを抜ける."""
        self.close()
