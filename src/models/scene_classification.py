"""Ollamaによる画像scene分類結果."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SceneClassification:
    """blog candidate 1枚のscene分類結果."""

    scene_slug: str
    scene_display_name: str
    scene_description: str
    confidence: float

    def __post_init__(self) -> None:
        """分類結果の妥当性を検証する."""
        if not self.scene_slug.strip():
            msg = "scene_slugは必須です"
            raise ValueError(msg)
        if not self.scene_display_name.strip():
            msg = "scene_display_nameは必須です"
            raise ValueError(msg)
        if not self.scene_description.strip():
            msg = "scene_descriptionは必須です"
            raise ValueError(msg)
        if not 0 <= self.confidence <= 1:
            msg = f"confidenceは0以上1以下である必要があります: {self.confidence}"
            raise ValueError(msg)
