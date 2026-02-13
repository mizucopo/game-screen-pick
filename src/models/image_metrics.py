"""Image metrics dataclass for storing analysis results."""

from dataclasses import dataclass

import numpy as np


@dataclass
class ImageMetrics:
    """画像解析結果を格納するデータクラス."""

    path: str
    raw_metrics: dict[str, float]
    normalized_metrics: dict[str, float]
    semantic_score: float
    total_score: float
    features: np.ndarray
