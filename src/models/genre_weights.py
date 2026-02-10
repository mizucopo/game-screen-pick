"""Genre weights for different game types."""

from typing import Dict


class GenreWeights:
    """全11ジャンルの重み定義."""

    DEFAULT_WEIGHTS = {
        "fps": {
            "blur_score": 0.25,
            "contrast": 0.20,
            "color_richness": 0.10,
            "visual_balance": 0.10,
            "edge_density": 0.10,
            "action_intensity": 0.15,
            "ui_density": 0.00,
            "dramatic_score": 0.10,
        },
        "tps": {
            "blur_score": 0.20,
            "contrast": 0.15,
            "color_richness": 0.15,
            "visual_balance": 0.15,
            "edge_density": 0.10,
            "action_intensity": 0.15,
            "ui_density": 0.00,
            "dramatic_score": 0.10,
        },
        "2d_action": {
            "blur_score": 0.15,
            "contrast": 0.15,
            "color_richness": 0.20,
            "visual_balance": 0.10,
            "edge_density": 0.05,
            "action_intensity": 0.15,
            "ui_density": 0.00,
            "dramatic_score": 0.20,
        },
        "2d_rpg": {
            "blur_score": 0.10,  # 低め - ピクセルアートは鮮明であるべき
            "contrast": 0.18,  # 高め - ピクセルアートのコントラスト重要
            "color_richness": 0.15,  # 中程度 - 限られたカラーパレット
            "visual_balance": 0.22,  # 高め - シンプルな構図のバランス
            "edge_density": 0.08,  # 低め - シンプルなスプライト
            "action_intensity": 0.05,  # 低め - ターン制でゆっくり
            "ui_density": 0.12,  # 高め - メニュー/テキストボックスが目立つ
            "dramatic_score": 0.10,  # 低め - シネマティックさは重要度低い
        },
        "3d_rpg": {
            "blur_score": 0.22,  # 高め - モーションブラー、被写界深度
            "contrast": 0.08,  # 低め - ソフトなライティング
            "color_richness": 0.25,  # 高め - リッチで詳細なビジュアル
            "visual_balance": 0.12,  # 低め - 複雑なシーン
            "edge_density": 0.15,  # 高め - 複雑で詳細なシーン
            "action_intensity": 0.08,  # 低め - ストーリー重視
            "ui_density": 0.02,  # 最小限 - 没入感重視
            "dramatic_score": 0.18,  # 高め - シネマティックな瞬間
        },
        "2d_shooting": {
            "blur_score": 0.20,
            "contrast": 0.20,
            "color_richness": 0.15,
            "visual_balance": 0.10,
            "edge_density": 0.05,
            "action_intensity": 0.10,
            "ui_density": 0.00,
            "dramatic_score": 0.20,
        },
        "3d_action": {
            "blur_score": 0.18,
            "contrast": 0.18,
            "color_richness": 0.18,
            "visual_balance": 0.12,
            "edge_density": 0.08,
            "action_intensity": 0.18,
            "ui_density": 0.00,
            "dramatic_score": 0.08,
        },
        "puzzle": {
            "blur_score": 0.25,
            "contrast": 0.20,
            "color_richness": 0.15,
            "visual_balance": 0.25,
            "edge_density": 0.10,
            "action_intensity": 0.00,
            "ui_density": 0.05,
            "dramatic_score": 0.00,
        },
        "racing": {
            "blur_score": 0.15,
            "contrast": 0.15,
            "color_richness": 0.15,
            "visual_balance": 0.15,
            "edge_density": 0.10,
            "action_intensity": 0.20,
            "ui_density": 0.00,
            "dramatic_score": 0.10,
        },
        "strategy": {
            "blur_score": 0.20,
            "contrast": 0.15,
            "color_richness": 0.15,
            "visual_balance": 0.20,
            "edge_density": 0.15,
            "action_intensity": 0.05,
            "ui_density": 0.10,
            "dramatic_score": 0.00,
        },
        "adventure": {
            "blur_score": 0.15,
            "contrast": 0.15,
            "color_richness": 0.20,
            "visual_balance": 0.20,
            "edge_density": 0.10,
            "action_intensity": 0.05,
            "ui_density": 0.00,
            "dramatic_score": 0.15,
        },
        "mixed": {
            "blur_score": 0.20,
            "contrast": 0.15,
            "color_richness": 0.15,
            "visual_balance": 0.15,
            "edge_density": 0.10,
            "action_intensity": 0.10,
            "ui_density": 0.05,
            "dramatic_score": 0.10,
        },
    }

    @classmethod
    def get_weights(cls, genre: str) -> Dict[str, float]:
        """ジャンルに対応する重みを取得."""
        return cls.DEFAULT_WEIGHTS.get(genre.lower(), cls.DEFAULT_WEIGHTS["mixed"])
