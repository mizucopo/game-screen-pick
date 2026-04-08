"""選定プロファイル定義。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SelectionProfile:
    """選定プロファイル定義."""

    name: str
    quality_weights: dict[str, float]

    def __post_init__(self) -> None:
        """ウェイト合計が1.0であることを検証する."""
        total = sum(self.quality_weights.values())
        if not (0.99 <= total <= 1.01):
            msg = (
                "quality_weightsの合計は1.0である必要があります"
                f"(許容誤差±0.01): {total}"
            )
            raise ValueError(msg)
