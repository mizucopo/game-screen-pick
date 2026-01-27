"""Genre weights for different game types."""

from typing import Dict


class GenreWeights:
    """全10ジャンルの重み定義."""
    DEFAULT_WEIGHTS = {
        "rpg": {
            "blur_score": 0.15, "contrast": 0.10, "color_richness": 0.20,
            "visual_balance": 0.15, "edge_density": 0.10,
            "action_intensity": 0.10, "ui_density": 0.05, "dramatic_score": 0.15
        },
        "fps": {
            "blur_score": 0.25, "contrast": 0.20, "color_richness": 0.10,
            "visual_balance": 0.10, "edge_density": 0.10,
            "action_intensity": 0.15, "ui_density": 0.00, "dramatic_score": 0.10
        },
        "tps": {
            "blur_score": 0.20, "contrast": 0.15, "color_richness": 0.15,
            "visual_balance": 0.15, "edge_density": 0.10,
            "action_intensity": 0.15, "ui_density": 0.00, "dramatic_score": 0.10
        },
        "2d_action": {
            "blur_score": 0.15, "contrast": 0.15, "color_richness": 0.20,
            "visual_balance": 0.10, "edge_density": 0.05,
            "action_intensity": 0.15, "ui_density": 0.00, "dramatic_score": 0.20
        },
        "2d_shooting": {
            "blur_score": 0.20, "contrast": 0.20, "color_richness": 0.15,
            "visual_balance": 0.10, "edge_density": 0.05,
            "action_intensity": 0.10, "ui_density": 0.00, "dramatic_score": 0.20
        },
        "3d_action": {
            "blur_score": 0.18, "contrast": 0.18, "color_richness": 0.18,
            "visual_balance": 0.12, "edge_density": 0.08,
            "action_intensity": 0.18, "ui_density": 0.00, "dramatic_score": 0.08
        },
        "puzzle": {
            "blur_score": 0.25, "contrast": 0.20, "color_richness": 0.15,
            "visual_balance": 0.25, "edge_density": 0.10,
            "action_intensity": 0.00, "ui_density": 0.05, "dramatic_score": 0.00
        },
        "racing": {
            "blur_score": 0.15, "contrast": 0.15, "color_richness": 0.15,
            "visual_balance": 0.15, "edge_density": 0.10,
            "action_intensity": 0.20, "ui_density": 0.00, "dramatic_score": 0.10
        },
        "strategy": {
            "blur_score": 0.20, "contrast": 0.15, "color_richness": 0.15,
            "visual_balance": 0.20, "edge_density": 0.15,
            "action_intensity": 0.05, "ui_density": 0.10, "dramatic_score": 0.00
        },
        "adventure": {
            "blur_score": 0.15, "contrast": 0.15, "color_richness": 0.20,
            "visual_balance": 0.20, "edge_density": 0.10,
            "action_intensity": 0.05, "ui_density": 0.00, "dramatic_score": 0.15
        },
        "mixed": {
            "blur_score": 0.20, "contrast": 0.15, "color_richness": 0.15,
            "visual_balance": 0.15, "edge_density": 0.10,
            "action_intensity": 0.10, "ui_density": 0.05, "dramatic_score": 0.10
        }
    }

    @classmethod
    def get_weights(cls, genre: str) -> Dict[str, float]:
        return cls.DEFAULT_WEIGHTS.get(genre.lower(), cls.DEFAULT_WEIGHTS["mixed"])
