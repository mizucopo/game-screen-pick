"""Image metrics dataclass for storing analysis results."""

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class ImageMetrics:
    """画像解析結果を格納するデータクラス."""

    path: str
    raw_metrics: Dict[str, float]
    normalized_metrics: Dict[str, float]
    semantic_score: float
    total_score: float
    features: np.ndarray
