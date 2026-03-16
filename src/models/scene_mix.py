"""Scene mix ratio model."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SceneMix:
    """画面種別ごとの選択比率."""

    play: float = 0.7
    event: float = 0.3

    def __post_init__(self) -> None:
        """比率の妥当性を検証する."""
        values = (self.play, self.event)
        if not all(0.0 <= value <= 1.0 for value in values):
            msg = f"scene_mixの各要素は0以上1以下である必要があります: {values}"
            raise ValueError(msg)
        total = sum(values)
        if not 0.99 <= total <= 1.01:
            msg = (
                "scene_mixの合計は1.0である必要があります"
                f"(許容誤差±0.01): {values} (合計: {total})"
            )
            raise ValueError(msg)

    def as_dict(self) -> dict[str, float]:
        """辞書形式で返す."""
        return {
            "play": self.play,
            "event": self.event,
        }
