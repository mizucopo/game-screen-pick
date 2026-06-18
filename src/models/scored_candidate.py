"""画面種別スコアと選定スコアを持つ候補。"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .analyzed_image import AnalyzedImage
from .scene_assessment import SceneAssessment
from .scene_selection_role import SceneSelectionRole


@dataclass
class ScoredCandidate:
    """最終選定に使うスコア付き候補."""

    analyzed_image: AnalyzedImage
    scene_assessment: SceneAssessment
    quality_score: float
    selection_score: float

    @property
    def path(self) -> str:
        """元画像パスを返す."""
        return self.analyzed_image.path

    @property
    def scene_label(self) -> str:
        """選定に使うscene slugを返す."""
        return self.scene_assessment.scene_slug

    @property
    def scene_slug(self) -> str:
        """scene slugを返す."""
        return self.scene_assessment.scene_slug

    @property
    def scene_display_name(self) -> str:
        """scene display nameを返す."""
        return self.scene_assessment.scene_display_name

    @property
    def scene_description(self) -> str:
        """scene descriptionを返す."""
        return self.scene_assessment.scene_description

    @property
    def scene_selection_role(self) -> SceneSelectionRole:
        """scene selection roleを返す."""
        return self.scene_assessment.scene_selection_role

    @property
    def combined_features(self) -> np.ndarray[Any, Any]:
        """類似度判定に使う結合特徴を返す."""
        return self.analyzed_image.combined_features
