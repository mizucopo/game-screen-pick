"""キャッシュエントリのデータ構造."""

from dataclasses import dataclass

import numpy as np


@dataclass
class CacheEntry:
    """キャッシュエントリのデータ構造.

    Attributes:
        clip_features: CLIP特徴（512次元ベクトル）
        raw_metrics: 生メトリクス（9つの値）
        hsv_features: HSV特徴（64次元ベクトル）
        semantic_score: セマンティックスコア
        normalized_metrics: 正規化メトリクス
        total_score: 総合スコア
    """

    clip_features: np.ndarray
    raw_metrics: dict[str, float]
    hsv_features: np.ndarray
    semantic_score: float
    normalized_metrics: dict[str, float]
    total_score: float
