"""画像品質アナライザーの設定."""

from dataclasses import dataclass


@dataclass
class AnalyzerConfig:
    """画像品質アナライザーの設定.

    Attributes:
        max_dim: メトリクス計算用の画像リサイズ時の長辺の最大ピクセル数
        max_memory_mb: チャンク処理時のメモリ予算（MB）。画像サイズ合計が
            この値を超えないように動的チャンク分割
        min_chunk_size: メモリ予算が大きい場合でも最低限確保するチャンクサイズ
        brightness_penalty_threshold: 輝度ペナルティを適用する輝度の境界値
        brightness_penalty_value: 輝度ペナルティの値
        semantic_weight: 総合スコア計算時のセマンティックスコアの重み
        score_multiplier: 総合スコアの乗数
    """

    max_dim: int = 720
    max_memory_mb: int = 512  # 約512MBのメモリ予算で動的チャンク
    min_chunk_size: int = 16  # 最低16枚は1チャンクで処理
    brightness_penalty_threshold: float = 40.0
    brightness_penalty_value: float = 0.6
    semantic_weight: float = 0.002  # コサイン類似度[-1,1]用に調整（元の0.2から100倍）
    score_multiplier: float = 100.0

    def __post_init__(self) -> None:
        """設定値の妥当性を検証する."""
        if self.max_dim <= 0:
            msg = f"max_dimは正の整数である必要があります: {self.max_dim}"
            raise ValueError(msg)
        if self.max_memory_mb <= 0:
            msg = f"max_memory_mbは正の整数である必要があります: {self.max_memory_mb}"
            raise ValueError(msg)
        if self.min_chunk_size <= 0:
            msg = f"min_chunk_sizeは正の整数である必要があります: {self.min_chunk_size}"
            raise ValueError(msg)
        if self.brightness_penalty_threshold < 0:
            msg = (
                "brightness_penalty_thresholdは非負の値である必要があります: "
                f"{self.brightness_penalty_threshold}"
            )
            raise ValueError(msg)
        if self.brightness_penalty_value < 0:
            msg = (
                "brightness_penalty_valueは非負の値である必要があります: "
                f"{self.brightness_penalty_value}"
            )
            raise ValueError(msg)
        if self.semantic_weight < 0:
            msg = (
                f"semantic_weightは非負の値である必要があります: {self.semantic_weight}"
            )
            raise ValueError(msg)
        if self.score_multiplier <= 0:
            msg = (
                f"score_multiplierは正の値である必要があります: {self.score_multiplier}"
            )
            raise ValueError(msg)
