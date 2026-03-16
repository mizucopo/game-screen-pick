"""選定時に使用する画面種別ラベル。"""

from enum import StrEnum


class SceneLabel(StrEnum):
    """選定時に扱う画面種別."""

    PLAY = "play"
    EVENT = "event"
