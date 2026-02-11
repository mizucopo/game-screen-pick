"""画像品質アナライザーの設定."""

from dataclasses import dataclass


@dataclass
class AnalyzerConfig:
    """画像品質アナライザーの設定.

    Attributes:
        max_dim: メトリクス計算用の画像リサイズ時の長辺の最大ピクセル数
        chunk_size: バッチ処理時のチャンクサイズ（メモリ使用量を抑える）
        brightness_penalty_threshold: 輝度ペナルティを適用する輝度の境界値
        brightness_penalty_value: 輝度ペナルティの値
        semantic_weight: 総合スコア計算時のセマンティックスコアの重み
        score_multiplier: 総合スコアの乗数
    """

    max_dim: int = 720
    chunk_size: int = 128
    brightness_penalty_threshold: float = 40.0
    brightness_penalty_value: float = 0.6
    semantic_weight: float = 0.2
    score_multiplier: float = 100.0
