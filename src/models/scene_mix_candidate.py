"""scene mix 選定が参照する候補インターフェース。"""

from typing import Any, Protocol

import numpy as np

from ..constants.scene_label import SceneLabel


class SceneMixCandidate(Protocol):
    """scene mix 選定に必要な候補情報."""

    @property
    def path(self) -> str:
        """元画像パスを返す."""
        ...

    @property
    def scene_label(self) -> SceneLabel:
        """scene mix 分類を返す."""
        ...

    @property
    def quality_score(self) -> float:
        """同一 score band 内の優先度を返す."""
        ...

    @property
    def selection_score(self) -> float:
        """score band 分割に使う選定スコアを返す."""
        ...

    @property
    def combined_features(self) -> np.ndarray[Any, Any]:
        """類似度判定に使う結合特徴を返す."""
        ...
