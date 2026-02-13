"""正規化されたメトリクス用データクラス."""

from dataclasses import dataclass


@dataclass(frozen=True)
class NormalizedMetrics:
    """正規化された画像メトリクス.

    Attributes:
        blur_score: 正規化されたぼけ score
        contrast: 正規化されたコントラスト
        color_richness: 正規化された彩度リッチネス
        edge_density: 正規化されたエッジ密度
        dramatic_score: 正規化されたドラマティックスコア
        visual_balance: 正規化された視覚的バランス
        action_intensity: 正規化されたアクション強度
        ui_density: 正規化されたUI密度
    """

    blur_score: float
    contrast: float
    color_richness: float
    edge_density: float
    dramatic_score: float
    visual_balance: float
    action_intensity: float
    ui_density: float

    def to_dict(self) -> dict[str, float]:
        """辞書に変換する（既存コードとの互換性用）.

        Returns:
            メトリクスの辞書
        """
        return {
            "blur_score": self.blur_score,
            "contrast": self.contrast,
            "color_richness": self.color_richness,
            "edge_density": self.edge_density,
            "dramatic_score": self.dramatic_score,
            "visual_balance": self.visual_balance,
            "action_intensity": self.action_intensity,
            "ui_density": self.ui_density,
        }

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> "NormalizedMetrics":
        """辞書から生成する（既存コードとの互換性用）.

        Args:
            data: メトリクスの辞書

        Returns:
            NormalizedMetricsインスタンス
        """
        return cls(
            blur_score=data["blur_score"],
            contrast=data["contrast"],
            color_richness=data["color_richness"],
            edge_density=data["edge_density"],
            dramatic_score=data["dramatic_score"],
            visual_balance=data["visual_balance"],
            action_intensity=data["action_intensity"],
            ui_density=data["ui_density"],
        )

    def get(self, key: str, default: float = 0.0) -> float:
        """辞書風のgetメソッド（既存コードとの互換性用）.

        Args:
            key: メトリクス名
            default: キーが存在しない場合のデフォルト値

        Returns:
            メトリクス値
        """
        return getattr(self, key, default)
