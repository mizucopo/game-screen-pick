"""Score weights for image quality scoring."""

from typing import Dict


class ScoreWeights:
    """画像品質スコアリングの重み定義.

    合計値は1.0に正規化されている。
    """

    DEFAULT_WEIGHTS: Dict[str, float] = {
        "blur_score": 0.15,
        "contrast": 0.14,
        "color_richness": 0.16,
        "visual_balance": 0.17,
        "edge_density": 0.11,
        "action_intensity": 0.12,
        "ui_density": 0.10,
        "dramatic_score": 0.05,
    }

    # 活動量計算用の重み（合計1.0）
    ACTIVITY_SCORE_WEIGHTS: Dict[str, float] = {
        "action_intensity": 0.55,
        "edge_density": 0.25,
        "dramatic_score": 0.20,
    }

    # 活動量ミックスのデフォルト設定
    DEFAULT_ACTIVITY_MIX_RATIO = (0.3, 0.4, 0.3)  # low, mid, high

    @classmethod
    def get_weights(cls) -> Dict[str, float]:
        """デフォルトの重みを取得.

        Returns:
            重みの辞書
        """
        return cls.DEFAULT_WEIGHTS.copy()

    @classmethod
    def get_activity_weights(cls) -> Dict[str, float]:
        """活動量計算用の重みを取得.

        Returns:
            活動量重みの辞書
        """
        return cls.ACTIVITY_SCORE_WEIGHTS.copy()
