"""分類対象pathを記録するテスト用scene analyzer。"""

import threading

from src.models.scene_catalog_entry import SceneCatalogEntry
from src.models.scene_classification import SceneClassification


class CountingSceneAnalyzer:
    """分類対象pathを記録するscene analyzer."""

    def __init__(self) -> None:
        """fake analyzerを初期化する."""
        self.catalog = [
            SceneCatalogEntry("battle", "戦闘", "敵と戦う場面"),
            SceneCatalogEntry("other", "その他", "分類しにくい場面"),
        ]
        self.representative_paths: list[str] = []
        self.classified_paths: list[str] = []
        self._lock = threading.Lock()

    def generate_scene_catalog(
        self,
        representative_paths: list[str],
        scene_hint: str | None,
    ) -> list[SceneCatalogEntry]:
        """scene catalogを返す."""
        assert scene_hint is None
        self.representative_paths = representative_paths
        return self.catalog

    def classify_image(
        self,
        image_path: str,
        catalog: list[SceneCatalogEntry],
    ) -> SceneClassification | None:
        """分類対象pathを記録して分類結果を返す."""
        assert catalog == self.catalog
        with self._lock:
            self.classified_paths.append(image_path)
        return SceneClassification(
            scene_slug="battle",
            scene_display_name="戦闘",
            scene_description="敵との戦闘場面",
            confidence=0.9,
        )
