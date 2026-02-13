"""画像選択の設定."""

from dataclasses import dataclass, field

from ..constants.score_weights import ScoreWeights
from .types.selection_config_kwargs import SelectionConfigKwargs


@dataclass
class SelectionConfig:
    """画像選択の設定.

    Attributes:
        batch_size: CLIP推論のバッチサイズ
        threshold_relaxation_steps: 類似度しきい値の段階的緩和ステップ
            （ベースしきい値に加算される値のリスト）
        max_threshold: 類似度しきい値の上限
        activity_mix_enabled: 活動量ミックスを有効にするかどうか
        activity_mix_ratio: 活動量バケットの選択比率 (low, mid, high)
        activity_bucket_mode: 活動量バケットの分割モード（"quantile"のみ）
    """

    batch_size: int = 32
    threshold_relaxation_steps: list[float] = field(
        default_factory=lambda: [0.03, 0.06, 0.10, 0.15]
    )
    max_threshold: float = 0.98
    activity_mix_enabled: bool = True
    activity_mix_ratio: tuple[float, float, float] = field(
        default_factory=lambda: ScoreWeights.DEFAULT_ACTIVITY_MIX_RATIO
    )
    activity_bucket_mode: str = "quantile"

    @classmethod
    def from_cli_args(cls, **kwargs: SelectionConfigKwargs) -> "SelectionConfig":
        """CLI引数から設定を作成する.

        Args:
            **kwargs: CLI引数（batch_size）
                Noneでない引数のみデフォルト値を上書き

        Returns:
            SelectionConfigインスタンス
        """
        filtered = {k: v for k, v in kwargs.items() if v is not None}
        return cls(**filtered)  # type: ignore[arg-type]

    def __post_init__(self) -> None:
        """設定値の妥当性を検証する."""
        if self.batch_size <= 0:
            msg = f"batch_sizeは正の整数である必要があります: {self.batch_size}"
            raise ValueError(msg)

        if not (0 <= self.max_threshold <= 1):
            msg = f"max_thresholdは0以上1以下である必要があります: {self.max_threshold}"
            raise ValueError(msg)

        for i, step in enumerate(self.threshold_relaxation_steps):
            if step < 0:
                msg = (
                    f"threshold_relaxation_steps[{i}]は非負の値である必要があります: "
                    f"{step}"
                )
                raise ValueError(msg)

        if self.activity_bucket_mode != "quantile":
            msg = (
                "activity_bucket_modeは'quantile'のみサポートしています: "
                f"{self.activity_bucket_mode}"
            )
            raise ValueError(msg)

        if len(self.activity_mix_ratio) != 3:
            msg = (
                "activity_mix_ratioは3要素のタプルである必要があります: "
                f"{self.activity_mix_ratio}"
            )
            raise ValueError(msg)

        if not all(0 <= r <= 1 for r in self.activity_mix_ratio):
            msg = (
                "activity_mix_ratioの各要素は0以上1以下である必要があります: "
                f"{self.activity_mix_ratio}"
            )
            raise ValueError(msg)

        total_ratio = sum(self.activity_mix_ratio)
        if not (0.99 <= total_ratio <= 1.01):
            msg = (
                "activity_mix_ratioの合計は1.0である必要です"
                f"(許容誤差±0.01): {self.activity_mix_ratio} (合計: {total_ratio})"
            )
            raise ValueError(msg)

    def compute_threshold_steps(self, base_threshold: float) -> list[float]:
        """ベースのしきい値から段階的な緩和ステップを計算する.

        Args:
            base_threshold: 基準となる類似度しきい値

        Returns:
            段階的に緩和されたしきい値のリスト（max_thresholdで上限付き）
        """
        steps = [base_threshold]
        for delta in self.threshold_relaxation_steps:
            next_threshold = min(base_threshold + delta, self.max_threshold)
            steps.append(next_threshold)
        return steps
