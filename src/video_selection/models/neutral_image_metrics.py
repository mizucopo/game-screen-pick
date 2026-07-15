"""model-free Neutral Image Analysisの画像metrics。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class NeutralImageMetrics:
    """画質、露出、情報量を表す数値metric。"""

    blur_score: float
    brightness: float
    contrast: float
    edge_density: float
    color_richness: float
    ui_density: float
    action_intensity: float
    visual_balance: float
    dramatic_score: float
    luminance_entropy: float
    luminance_range: float
    near_black_ratio: float
    near_white_ratio: float
    dominant_tone_ratio: float
    information_score: float
    visibility_score: float
