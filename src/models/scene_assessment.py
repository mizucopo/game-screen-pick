"""scene分類から導出される画面評価。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SceneAssessment:
    """画面種別の評価結果."""

    scene_slug: str = ""
    scene_display_name: str = ""
    scene_description: str = ""
    scene_confidence: float = 0.0

    def __post_init__(self) -> None:
        """表示名と説明を補完する."""
        if not self.scene_display_name:
            object.__setattr__(self, "scene_display_name", self.scene_slug)
        if not self.scene_description:
            object.__setattr__(
                self,
                "scene_description",
                self.scene_display_name,
            )
