"""Genre weights for different game types."""

from typing import Dict


class GenreWeights:
    """ジャンル別の重み定義."""

    DEFAULT_WEIGHTS = {
        "fps": {
            "blur_score": 0.22,
            "contrast": 0.19,
            "color_richness": 0.09,
            "visual_balance": 0.08,
            "edge_density": 0.15,
            "action_intensity": 0.18,
            "ui_density": 0.03,
            "dramatic_score": 0.06,
        },
        "tps": {
            "blur_score": 0.19,
            "contrast": 0.16,
            "color_richness": 0.14,
            "visual_balance": 0.14,
            "edge_density": 0.12,
            "action_intensity": 0.15,
            "ui_density": 0.03,
            "dramatic_score": 0.07,
        },
        "2d_action": {
            "blur_score": 0.17,
            "contrast": 0.16,
            "color_richness": 0.20,
            "visual_balance": 0.09,
            "edge_density": 0.07,
            "action_intensity": 0.16,
            "ui_density": 0.03,
            "dramatic_score": 0.12,
        },
        "2d_rpg": {
            "blur_score": 0.11,
            "contrast": 0.17,
            "color_richness": 0.16,
            "visual_balance": 0.21,
            "edge_density": 0.07,
            "action_intensity": 0.05,
            "ui_density": 0.16,
            "dramatic_score": 0.07,
        },
        "3d_rpg": {
            "blur_score": 0.18,
            "contrast": 0.10,
            "color_richness": 0.22,
            "visual_balance": 0.12,
            "edge_density": 0.13,
            "action_intensity": 0.07,
            "ui_density": 0.04,
            "dramatic_score": 0.14,
        },
        "2d_shooting": {
            "blur_score": 0.18,
            "contrast": 0.19,
            "color_richness": 0.15,
            "visual_balance": 0.08,
            "edge_density": 0.08,
            "action_intensity": 0.17,
            "ui_density": 0.03,
            "dramatic_score": 0.12,
        },
        "3d_action": {
            "blur_score": 0.17,
            "contrast": 0.18,
            "color_richness": 0.17,
            "visual_balance": 0.11,
            "edge_density": 0.12,
            "action_intensity": 0.18,
            "ui_density": 0.02,
            "dramatic_score": 0.05,
        },
        "puzzle": {
            "blur_score": 0.22,
            "contrast": 0.19,
            "color_richness": 0.14,
            "visual_balance": 0.26,
            "edge_density": 0.08,
            "action_intensity": 0.02,
            "ui_density": 0.08,
            "dramatic_score": 0.01,
        },
        "racing": {
            "blur_score": 0.14,
            "contrast": 0.16,
            "color_richness": 0.14,
            "visual_balance": 0.13,
            "edge_density": 0.11,
            "action_intensity": 0.23,
            "ui_density": 0.04,
            "dramatic_score": 0.05,
        },
        "strategy": {
            "blur_score": 0.16,
            "contrast": 0.14,
            "color_richness": 0.12,
            "visual_balance": 0.21,
            "edge_density": 0.17,
            "action_intensity": 0.04,
            "ui_density": 0.14,
            "dramatic_score": 0.02,
        },
        "adventure": {
            "blur_score": 0.14,
            "contrast": 0.14,
            "color_richness": 0.22,
            "visual_balance": 0.20,
            "edge_density": 0.10,
            "action_intensity": 0.06,
            "ui_density": 0.03,
            "dramatic_score": 0.11,
        },
        "mixed": {
            "blur_score": 0.17,
            "contrast": 0.15,
            "color_richness": 0.15,
            "visual_balance": 0.15,
            "edge_density": 0.12,
            "action_intensity": 0.11,
            "ui_density": 0.06,
            "dramatic_score": 0.09,
        },
    }

    @classmethod
    def get_weights(cls, genre: str) -> Dict[str, float]:
        """ジャンルに対応する重みを取得."""
        return cls.DEFAULT_WEIGHTS.get(genre.lower(), cls.DEFAULT_WEIGHTS["mixed"])
