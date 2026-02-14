"""Image metrics dataclass for storing analysis results."""

from dataclasses import dataclass

import numpy as np

from .normalized_metrics import NormalizedMetrics
from .raw_metrics import RawMetrics


@dataclass
class ImageMetrics:
    """画像解析結果を格納するデータクラス.

    Attributes:
        path: 画像ファイルパス
        raw_metrics: 生メトリクス
        normalized_metrics: 正規化済みメトリクス
        semantic_score: セマンティック類似度スコア（CLIPモデルによる）
        total_score: 総合品質スコア（重み付き合計）
        features: 統合特徴ベクトル（HSV + CLIP、正規化済み）
    """

    path: str
    raw_metrics: RawMetrics
    normalized_metrics: NormalizedMetrics
    semantic_score: float
    total_score: float
    features: np.ndarray
