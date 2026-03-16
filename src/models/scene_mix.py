"""画面種別ミックス比率モデル。"""

from dataclasses import dataclass

from ..constants.scene_label import SceneLabel


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

    def calculate_allocation(self, total: int) -> dict[SceneLabel, int]:
        """総数から play/event の配分を計算する.

        Args:
            total: 配分する総数。

        Returns:
            SceneLabelごとの配分数。
        """
        raw_play = total * self.play
        raw_event = total * self.event
        play_target = int(raw_play)
        event_target = int(raw_event)
        remainder = total - (play_target + event_target)
        if remainder > 0 and raw_play - play_target >= raw_event - event_target:
            play_target += 1
        elif remainder > 0:
            event_target += 1
        return {SceneLabel.PLAY: play_target, SceneLabel.EVENT: event_target}
