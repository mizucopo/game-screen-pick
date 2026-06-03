"""テスト用scene analyzer."""

from src.models.scene_catalog_entry import SceneCatalogEntry
from src.models.scene_classification import SceneClassification


class FakeSceneAnalyzer:
    """固定のscene catalogと分類結果を返すfake."""

    def __init__(
        self,
        classifications_by_path: dict[str, SceneClassification] | None = None,
        default_scene_slug: str = "battle",
    ) -> None:
        """fake scene analyzerを初期化する."""
        self.catalog = [
            SceneCatalogEntry("battle", "戦闘", "敵と戦う場面"),
            SceneCatalogEntry("conversation", "会話", "人物同士の会話場面"),
            SceneCatalogEntry("other", "その他", "分類しにくい場面"),
        ]
        self.classifications_by_path = classifications_by_path or {}
        self.default_scene_slug = default_scene_slug

    def generate_scene_catalog(
        self,
        representative_paths: list[str],
        scene_hint: str | None,
    ) -> list[SceneCatalogEntry]:
        """scene catalogを返す."""
        del representative_paths, scene_hint
        return self.catalog

    def classify_image(
        self,
        image_path: str,
        catalog: list[SceneCatalogEntry],
    ) -> SceneClassification | None:
        """pathに対応する分類結果またはdefault分類を返す."""
        del catalog
        if image_path in self.classifications_by_path:
            return self.classifications_by_path[image_path]
        scene = next(
            item for item in self.catalog if item.slug == self.default_scene_slug
        )
        return SceneClassification(
            scene_slug=scene.slug,
            scene_display_name=scene.display_name,
            scene_description=scene.description,
            confidence=0.8,
        )
