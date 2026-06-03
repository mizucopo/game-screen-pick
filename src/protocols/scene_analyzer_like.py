"""scene分析adapterのProtocol."""

from typing import Protocol

from ..models.scene_catalog_entry import SceneCatalogEntry
from ..models.scene_classification import SceneClassification


class SceneAnalyzerLike(Protocol):
    """scene catalog作成と画像分類を行う境界."""

    def generate_scene_catalog(
        self,
        representative_paths: list[str],
        scene_hint: str | None,
    ) -> list[SceneCatalogEntry]:
        """代表画像からscene catalogを作成する."""

    def classify_image(
        self,
        image_path: str,
        catalog: list[SceneCatalogEntry],
    ) -> SceneClassification | None:
        """画像をscene catalogのsceneへ分類する."""
