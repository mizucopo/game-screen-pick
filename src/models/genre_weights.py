"""Genre weights for different game types."""

from typing import Dict


class GenreWeights:
    """全12ジャンルの重み定義."""

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
            "blur_score": 0.11,  # 低め - ピクセルアートは鮮明であるべき
            "contrast": 0.17,  # 高め - ピクセルアートのコントラスト重要
            "color_richness": 0.16,  # 中程度 - 限られたカラーパレット
            "visual_balance": 0.21,  # 高め - シンプルな構図のバランス
            "edge_density": 0.07,  # 低め - シンプルなスプライト
            "action_intensity": 0.05,  # 低め - ターン制でゆっくり
            "ui_density": 0.16,  # 高め - メニュー/テキストボックスが目立つ
            "dramatic_score": 0.07,  # 低め - シネマティックさは重要度低い
        },
        "3d_rpg": {
            "blur_score": 0.18,  # やや高め - 高解像度シーンの鮮明さ重視
            "contrast": 0.10,  # 低め - ソフトなライティング
            "color_richness": 0.22,  # 高め - リッチで詳細なビジュアル
            "visual_balance": 0.12,  # 中程度 - 広いシーンの構図
            "edge_density": 0.13,  # 高め - 複雑で詳細なシーン
            "action_intensity": 0.07,  # 低め - ストーリー重視
            "ui_density": 0.04,  # 低め - 没入感重視
            "dramatic_score": 0.14,  # 高め - シネマティックな瞬間
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
