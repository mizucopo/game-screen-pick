"""ブログ画像選定で使う品質スコア重み。"""

DEFAULT_QUALITY_WEIGHTS = {
    "blur_score": 0.165,
    "contrast": 0.145,
    "color_richness": 0.06,
    "visual_balance": 0.10,
    "edge_density": 0.13,
    "action_intensity": 0.125,
    "ui_density": 0.225,
    "dramatic_score": 0.05,
}
