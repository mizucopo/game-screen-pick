"""正規化されたメトリクス用データクラス."""

from dataclasses import dataclass


@dataclass(frozen=True)
class NormalizedMetrics:
    """正規化された画像メトリクス（全て0.0-1.0の範囲）.

    Attributes:
        blur_score: 正規化されたぼけ度スコア（シグモイド関数適用）
        contrast: 正規化されたコントラスト（シグモイド関数適用）
        color_richness: 正規化された彩度リッチネス（シグモイド関数適用）
        edge_density: 正規化されたエッジ密度（線形スケーリング）
        dramatic_score: 正規化されたドラマティックスコア（線形スケーリング）
        visual_balance: 正規化された視覚的バランス（線形スケーリング）
        action_intensity: 正規化されたアクション強度（シグモイド関数適用）
        ui_density: 正規化されたUI密度（シグモイド関数適用）
    """

    blur_score: float
    contrast: float
    color_richness: float
    edge_density: float
    dramatic_score: float
    visual_balance: float
    action_intensity: float
    ui_density: float

    def __post_init__(self) -> None:
        """全フィールドが0.0〜1.0の範囲内であることを検証する."""
        for field_name in (
            "blur_score",
            "contrast",
            "color_richness",
            "edge_density",
            "dramatic_score",
            "visual_balance",
            "action_intensity",
            "ui_density",
        ):
            value = getattr(self, field_name)
            if not (0.0 <= value <= 1.0):
                msg = f"{field_name}は0.0〜1.0の範囲である必要があります: {value}"
                raise ValueError(msg)
