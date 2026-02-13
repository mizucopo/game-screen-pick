"""Image metrics dataclass for storing analysis results."""

from dataclasses import dataclass

import numpy as np

from .normalized_metrics import NormalizedMetrics
from .raw_metrics import RawMetrics


@dataclass
class ImageMetrics:
    """画像解析結果を格納するデータクラス."""

    path: str
    raw_metrics: RawMetrics
    normalized_metrics: NormalizedMetrics
    semantic_score: float
    total_score: float
    features: np.ndarray
