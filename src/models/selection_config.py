"""画像選択の設定."""

from dataclasses import dataclass, field

from ..constants.selection_profiles import DEFAULT_SCENE_MIX
from .config_from_args_mixin import ConfigFromArgsMixin
from .scene_mix import SceneMix


@dataclass
class SelectionConfig(ConfigFromArgsMixin):
    """画像選択の設定.

    Attributes:
        batch_size: CLIP推論のバッチサイズ
        profile: 実行プロファイル（auto / active / static）
        similarity_threshold: 類似度しきい値
        scene_mix: 画面種別ごとの選択比率
        threshold_relaxation_steps: 類似度しきい値の段階的緩和ステップ
            （ベースしきい値に加算される値のリスト）
        max_threshold: 類似度しきい値の上限
    """

    batch_size: int = 32
    profile: str = "auto"
    similarity_threshold: float = 0.72
    scene_mix: SceneMix = field(default_factory=lambda: DEFAULT_SCENE_MIX)
    threshold_relaxation_steps: list[float] = field(
        default_factory=lambda: [0.03, 0.06, 0.10, 0.15]
    )
    max_threshold: float = 0.98

    def __post_init__(self) -> None:
        """設定値の妥当性を検証する."""
        if self.batch_size <= 0:
            msg = f"batch_sizeは正の整数である必要があります: {self.batch_size}"
            raise ValueError(msg)

        if self.profile not in {"auto", "active", "static"}:
            msg = (
                "profileは'auto', 'active', 'static'のいずれかである必要があります: "
                f"{self.profile}"
            )
            raise ValueError(msg)

        if not (0 <= self.similarity_threshold <= 1):
            msg = (
                "similarity_thresholdは0以上1以下である必要があります: "
                f"{self.similarity_threshold}"
            )
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
