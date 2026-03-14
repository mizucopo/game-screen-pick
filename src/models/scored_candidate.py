"""Candidate enriched with scene and selection scores."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .analyzed_image import AnalyzedImage
from .scene_assessment import SceneAssessment


@dataclass(frozen=True)
class ScoredCandidate:
    """最終選定に使うスコア付き候補."""

    analyzed_image: AnalyzedImage
    scene_assessment: SceneAssessment
    resolved_profile: str
    quality_score: float
    activity_score: float
    selection_score: float

    @property
    def path(self) -> str:
        """元画像パスを返す."""
        return self.analyzed_image.path

    @property
    def combined_features(self) -> np.ndarray[Any, Any]:
        """類似度判定に使う結合特徴を返す."""
        return self.analyzed_image.combined_features
