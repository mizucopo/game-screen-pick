"""画面種別スコアと選定スコアを持つ候補。"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..constants.scene_label import SceneLabel
from .analyzed_image import AnalyzedImage
from .scene_assessment import SceneAssessment


@dataclass
class ScoredCandidate:
    """最終選定に使うスコア付き候補."""

    analyzed_image: AnalyzedImage
    scene_assessment: SceneAssessment
    resolved_profile: str
    quality_score: float
    selection_score: float

    @property
    def path(self) -> str:
        """元画像パスを返す."""
        return self.analyzed_image.path

    @property
    def scene_label(self) -> SceneLabel:
        """scene mix 選定に使う画面種別を返す."""
        return self.scene_assessment.scene_label

    @property
    def combined_features(self) -> np.ndarray[Any, Any]:
        """類似度判定に使う結合特徴を返す."""
        return self.analyzed_image.combined_features
