"""画像品質アナライザーの設定."""

from dataclasses import dataclass

from .config_from_args_mixin import ConfigFromArgsMixin


@dataclass
class AnalyzerConfig(ConfigFromArgsMixin):
    """画像品質アナライザーの設定.

    Attributes:
        max_dim: メトリクス計算用の画像リサイズ時の長辺の最大ピクセル数
        max_memory_gb: チャンク処理時のメモリ予算（GB）。画像サイズ合計が
            この値を超えないように動的チャンク分割
        min_chunk_size: メモリ予算が大きい場合でも最低限確保するチャンクサイズ
        brightness_penalty_threshold: 輝度ペナルティを適用する輝度の境界値
        brightness_penalty_value: 輝度ペナルティの値
        score_multiplier: 総合スコアの乗数
        result_max_workers: 結果構築（raw metric + feature結合）の並列処理ワーカー数
            Noneでデフォルト値（min(8, max(1, os.cpu_count() - 1))）を使用
        io_max_workers: 画像読み込みI/Oの並列処理ワーカー数。Noneで自動設定
    """

    max_dim: int = 720
    max_memory_gb: int = 1  # 約1GBのメモリ予算で動的チャンク
    min_chunk_size: int = 16  # 最低16枚は1チャンクで処理
    brightness_penalty_threshold: float = 35.0
    brightness_penalty_value: float = 0.15
    score_multiplier: float = 100.0
    result_max_workers: int | None = None
    io_max_workers: int | None = None

    @staticmethod
    def _validate_positive(name: str, value: int | float) -> None:
        if value < 1:
            msg = f"{name}は1以上の値を指定してください: {value}"
            raise ValueError(msg)

    @staticmethod
    def _validate_non_negative(name: str, value: int | float) -> None:
        if value < 0:
            msg = f"{name}は0以上の値を指定してください: {value}"
            raise ValueError(msg)

    def __post_init__(self) -> None:
        """設定値の妥当性を検証する."""
        self._validate_positive("max_dim", self.max_dim)
        self._validate_positive("max_memory_gb", self.max_memory_gb)
        self._validate_positive("min_chunk_size", self.min_chunk_size)
        self._validate_non_negative(
            "brightness_penalty_threshold",
            self.brightness_penalty_threshold,
        )
        self._validate_non_negative(
            "brightness_penalty_value",
            self.brightness_penalty_value,
        )
        self._validate_positive("score_multiplier", self.score_multiplier)
        if self.result_max_workers is not None:
            self._validate_non_negative("result_max_workers", self.result_max_workers)
        if self.io_max_workers is not None:
            self._validate_non_negative("io_max_workers", self.io_max_workers)
