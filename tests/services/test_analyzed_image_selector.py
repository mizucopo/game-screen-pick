"""AnalyzedImageSelector の単体テスト."""

from src.analyzers.metric_calculator import MetricCalculator
from src.models.analyzer_config import AnalyzerConfig
from src.models.ollama_config import OllamaConfig
from src.models.scene_catalog_entry import SceneCatalogEntry
from src.models.scene_classification import SceneClassification
from src.models.selection_config import SelectionConfig
from src.services.analyzed_image_selector import AnalyzedImageSelector
from tests.conftest import _feature, create_analyzed_image


def test_select_classifies_blog_candidates_and_reports_ollama_failures() -> None:
    """解析済み画像から選定結果と統計が生成されること.

    Arrange:
        - content filter で除外される画像と選定対象の画像がある
        - fake scene analyzer が2枚を分類し1枚は分類失敗にする
    Act:
        - AnalyzedImageSelectorで選定される
    Assert:
        - 除外画像を含まない選定結果が返されること
        - 選定統計にscene catalogとOllama分類失敗が反映されること
    """
    # Arrange
    dark = create_analyzed_image(
        path="/tmp/dark.jpg",
        raw_metrics_dict={
            "near_black_ratio": 0.98,
            "luminance_entropy": 0.2,
            "luminance_range": 10.0,
        },
        combined_features=_feature(0),
    )
    battle = create_analyzed_image(
        path="/tmp/battle.jpg",
        combined_features=_feature(1),
    )
    conversation = create_analyzed_image(
        path="/tmp/conversation.jpg",
        combined_features=_feature(100),
    )
    failure = create_analyzed_image(
        path="/tmp/failure.jpg",
        combined_features=_feature(200),
    )
    scene_analyzer = _FakeSceneAnalyzer(
        classifications_by_path={
            "/tmp/battle.jpg": SceneClassification(
                scene_slug="battle",
                scene_display_name="戦闘",
                scene_description="敵との戦闘場面",
                confidence=0.9,
            ),
            "/tmp/conversation.jpg": SceneClassification(
                scene_slug="conversation",
                scene_display_name="会話",
                scene_description="人物同士の会話場面",
                confidence=0.8,
            ),
        }
    )
    selector = AnalyzedImageSelector(
        config=SelectionConfig(
            profile="active",
            ollama=OllamaConfig(model="gemma4"),
        ),
        metric_calculator=MetricCalculator(AnalyzerConfig()),
        scene_analyzer=scene_analyzer,
    )

    # Act
    selected, rejected, stats = selector.select(
        analyzed_images=[dark, battle, conversation, failure],
        num=2,
        total_files=5,
        analyzed_fail=1,
    )

    # Assert
    assert {candidate.path for candidate in selected} == {
        "/tmp/battle.jpg",
        "/tmp/conversation.jpg",
    }
    assert rejected == []
    assert stats.total_files == 5
    assert stats.analyzed_ok == 4
    assert stats.analyzed_fail == 1
    assert stats.selected_count == 2
    assert stats.rejected_by_content_filter == 1
    assert stats.content_filter_breakdown["blackout"] == 1
    assert stats.scene_distribution == {"battle": 1, "conversation": 1}
    assert stats.scene_mix_target == {"battle": 1, "conversation": 1}
    assert stats.scene_mix_actual == {"battle": 1, "conversation": 1}
    assert stats.ollama_classification_failed == 1
    assert stats.ollama_classification_failure_rate == 1 / 3
    assert stats.scene_catalog[0].slug == "battle"
    assert stats.resolved_profile == "active"


class _FakeSceneAnalyzer:
    """テスト用のscene analyzer."""

    def __init__(
        self,
        classifications_by_path: dict[str, SceneClassification],
    ) -> None:
        """fake analyzerを初期化する."""
        self.classifications_by_path = classifications_by_path
        self.catalog = [
            SceneCatalogEntry("battle", "戦闘", "敵と戦う場面"),
            SceneCatalogEntry("conversation", "会話", "人物同士の会話場面"),
            SceneCatalogEntry("other", "その他", "分類しにくい場面"),
        ]

    def generate_scene_catalog(
        self,
        representative_paths: list[str],
        scene_hint: str | None,
    ) -> list[SceneCatalogEntry]:
        """scene catalogを返す."""
        assert "/tmp/dark.jpg" not in representative_paths
        assert scene_hint is None
        return self.catalog

    def classify_image(
        self,
        image_path: str,
        catalog: list[SceneCatalogEntry],
    ) -> SceneClassification | None:
        """事前設定した分類結果を返す."""
        assert catalog == self.catalog
        return self.classifications_by_path.get(image_path)
