"""scene selection role."""

from enum import StrEnum


class SceneSelectionRole(StrEnum):
    """sceneごとの最終選択での扱い."""

    ORDINARY = "ordinary"
    CINEMATIC = "cinematic"
    RECURRING_GAMEPLAY = "recurring_gameplay"

    @classmethod
    def from_value(cls, value: object) -> "SceneSelectionRole":
        """未知の値をordinaryへ正規化してroleを返す."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value.strip())
            except ValueError:
                return cls.ORDINARY
        return cls.ORDINARY
