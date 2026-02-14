"""生メトリクス用の型付きデータクラス."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RawMetrics:
    """生の画像メトリクス（画像解析から直接得られた値）.

    Attributes:
        blur_score: ぼけ度スコア（Laplacian分散）
        brightness: 輝度（平均輝度、0-255）
        contrast: コントラスト（輝度の標準偏差）
        edge_density: エッジ密度（Cannyエッジのピクセル比率）
        color_richness: 色彩度リッチネス（彩度の標準偏差）
        ui_density: UI密度（Sobel横方向エッジ強度）
        action_intensity: アクション強度（カーネル適用後の標準偏差）
        visual_balance: 視覚的バランス（輝度が128に近いほど高スコア）
        dramatic_score: ドラマティックスコア（高彩度・高輝度ピクセル比率）
    """

    blur_score: float
    brightness: float
    contrast: float
    edge_density: float
    color_richness: float
    ui_density: float
    action_intensity: float
    visual_balance: float
    dramatic_score: float
