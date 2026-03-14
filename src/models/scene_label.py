"""Scene bucket labels used during selection."""

from enum import StrEnum


class SceneLabel(StrEnum):
    """選定時に扱う画面種別."""

    GAMEPLAY = "gameplay"
    EVENT = "event"
    OTHER = "other"
