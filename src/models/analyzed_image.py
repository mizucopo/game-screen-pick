"""Neutral analysis result before scene-specific scoring."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .layout_heuristics import LayoutHeuristics
from .normalized_metrics import NormalizedMetrics
from .raw_metrics import RawMetrics


@dataclass(frozen=True)
class AnalyzedImage:
    """画像の中立解析結果."""

    path: str
    raw_metrics: RawMetrics
    normalized_metrics: NormalizedMetrics
    clip_features: np.ndarray[Any, Any]
    combined_features: np.ndarray[Any, Any]
    layout_heuristics: LayoutHeuristics
