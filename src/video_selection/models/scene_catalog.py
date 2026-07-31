"""Video Setを横断して共有するScene Catalog。"""

from dataclasses import dataclass

from .scene_catalog_entry import SceneCatalogEntry


@dataclass(frozen=True)
class SceneCatalog:
    """3〜8件の重複しないsceneと分類先otherを保持する。"""

    scenes: tuple[SceneCatalogEntry, ...]

    def __post_init__(self) -> None:
        """scene数、slug一意性、otherのkindとroleを検証する。"""
        slugs = tuple(scene.slug for scene in self.scenes)
        other_scenes = tuple(scene for scene in self.scenes if scene.slug == "other")
        if (
            not 3 <= len(self.scenes) <= 8
            or len(slugs) != len(set(slugs))
            or len(other_scenes) != 1
            or other_scenes[0].scene_kind != "other"
            or other_scenes[0].selection_role != "ordinary"
        ):
            msg = "Scene Catalogのscene数、slug、other kind・roleが不正です"
            raise ValueError(msg)

    @property
    def slugs(self) -> tuple[str, ...]:
        """Catalog順のScene Slugを返す。"""
        return tuple(scene.slug for scene in self.scenes)

    def for_slug(self, slug: str) -> SceneCatalogEntry:
        """指定slugのsceneを返す。"""
        try:
            return next(scene for scene in self.scenes if scene.slug == slug)
        except StopIteration:
            raise KeyError(slug) from None
