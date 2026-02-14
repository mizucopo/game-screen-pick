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
