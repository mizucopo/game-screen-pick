"""画像選択の設定."""

from dataclasses import dataclass, field


@dataclass
class SelectionConfig:
    """画像選択の設定.

    Attributes:
        batch_size: バッチ処理のバッチサイズ
        threshold_relaxation_steps: 類似度しきい値の段階的緩和ステップ
            （ベースのしきい値に加算される値のリスト）
        max_threshold: 類似度しきい値の上限
    """

    batch_size: int = 32
    threshold_relaxation_steps: list[float] = field(
        default_factory=lambda: [0.03, 0.06, 0.10, 0.15]
    )
    max_threshold: float = 0.98

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
