"""Scene Catalog推論へ渡すsemantic入力。"""

from dataclasses import dataclass

from .frame_candidate import FrameCandidate


@dataclass(frozen=True)
class SceneCatalogRequest:
    """最大24枚の代表画像とSelection Intentを保持する。"""

    representatives: tuple[FrameCandidate, ...]
    selection_intent: str
    scene_hint: str | None = None

    def __post_init__(self) -> None:
        """代表画像数、内容、識別子、意図を検証する。"""
        identifiers = tuple(item.identifier for item in self.representatives)
        if (
            not 1 <= len(self.representatives) <= 24
            or len(identifiers) != len(set(identifiers))
            or any(not item.image_bytes for item in self.representatives)
            or not self.selection_intent.strip()
            or self.scene_hint is not None
            and not self.scene_hint.strip()
        ):
            msg = "Scene Catalog requestが不正です"
            raise ValueError(msg)
