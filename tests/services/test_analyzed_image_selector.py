"""AnalyzedImageSelector の単体テスト."""

import threading
import time
from unittest.mock import patch

import numpy as np

from src.analyzers.metric_calculator import MetricCalculator
from src.models.analyzer_config import AnalyzerConfig
from src.models.ollama_config import OllamaConfig
from src.models.scene_catalog_entry import SceneCatalogEntry
from src.models.scene_classification import SceneClassification
from src.models.scene_selection_role import SceneSelectionRole
from src.models.selection_config import SelectionConfig
from src.services.analyzed_image_selector import AnalyzedImageSelector
from tests.conftest import _feature, create_analyzed_image
from tests.counting_scene_analyzer import CountingSceneAnalyzer
from tests.fake_scene_analyzer import FakeSceneAnalyzer


def _similar_feature(
    *,
    common_index: int,
    unique_index: int,
    dim: int,
    common_value: float,
    unique_value: float,
) -> np.ndarray:
    """共通成分を持つ類似特徴ベクトルを生成する."""
    feature = np.zeros(dim, dtype=np.float32)
    feature[common_index] = common_value
    feature[unique_index] = unique_value
    return feature


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
    roles_by_path = {
        candidate.path: candidate.scene_selection_role for candidate in selected
    }
    assert roles_by_path == {
        "/tmp/battle.jpg": SceneSelectionRole.RECURRING_GAMEPLAY,
        "/tmp/conversation.jpg": SceneSelectionRole.CINEMATIC,
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


def test_select_classifies_images_with_configured_ollama_workers() -> None:
    """設定されたOllama worker数で画像分類が並列実行されること.

    Arrange:
        - ollama.max_workersが2に設定されている
        - 複数の選定対象画像がある
    Act:
        - AnalyzedImageSelectorで選定される
    Assert:
        - 同時分類数が2以上になること
    """
    # Arrange
    first = create_analyzed_image(
        path="/tmp/first.jpg",
        combined_features=_feature(1),
    )
    second = create_analyzed_image(
        path="/tmp/second.jpg",
        combined_features=_feature(100),
    )
    third = create_analyzed_image(
        path="/tmp/third.jpg",
        combined_features=_feature(200),
    )
    scene_analyzer = _ConcurrentSceneAnalyzer()
    selector = AnalyzedImageSelector(
        config=SelectionConfig(
            ollama=OllamaConfig(model="gemma4", max_workers=2),
        ),
        metric_calculator=MetricCalculator(AnalyzerConfig()),
        scene_analyzer=scene_analyzer,
    )

    # Act
    selector.select(
        analyzed_images=[first, second, third],
        num=2,
    )

    # Assert
    assert scene_analyzer.max_active >= 2


def test_select_logs_ollama_progress() -> None:
    """Ollamaによるcatalog作成と画像分類の進捗が出力されること.

    Arrange:
        - 複数の選定対象画像がある
        - fake scene analyzer が全画像を分類する
    Act:
        - AnalyzedImageSelectorで選定される
    Assert:
        - catalog作成と画像分類の進捗ログが出力されること
    """
    # Arrange
    first = create_analyzed_image(
        path="/tmp/first.jpg",
        combined_features=_feature(1),
    )
    second = create_analyzed_image(
        path="/tmp/second.jpg",
        combined_features=_feature(100),
    )
    scene_analyzer = _FakeSceneAnalyzer(
        classifications_by_path={
            "/tmp/first.jpg": SceneClassification(
                scene_slug="battle",
                scene_display_name="戦闘",
                scene_description="敵との戦闘場面",
                confidence=0.9,
            ),
            "/tmp/second.jpg": SceneClassification(
                scene_slug="conversation",
                scene_display_name="会話",
                scene_description="人物同士の会話場面",
                confidence=0.8,
            ),
        }
    )
    selector = AnalyzedImageSelector(
        config=SelectionConfig(
            ollama=OllamaConfig(model="gemma4", max_workers=2),
        ),
        metric_calculator=MetricCalculator(AnalyzerConfig()),
        scene_analyzer=scene_analyzer,
    )

    # Act
    with patch("src.services.analyzed_image_selector.logger") as logger:
        selector.select(
            analyzed_images=[first, second],
            num=2,
        )

    # Assert
    messages = [call.args[0] for call in logger.info.call_args_list]
    assert "Ollama scene catalog作成中: 代表画像 2 件" in messages
    assert "Ollama scene catalog作成完了: scene 3 件" in messages
    assert "Ollama画像分類開始: 対象 2 件, worker 2" in messages
    assert "Ollama画像分類進捗: 1/2" in messages
    assert "Ollama画像分類進捗: 2/2" in messages
    assert "Ollama画像分類完了: 成功 2 件, 失敗 0 件" in messages


def test_select_classifies_selection_shortlist_instead_of_all_blog_candidates() -> None:
    """Selection ShortlistだけがOllama分類へ進められること.

    Arrange:
        - 500件を超えるblog candidateがある
        - すべての画像が分類可能である
    Act:
        - AnalyzedImageSelectorで少数枚が選定される
    Assert:
        - 全blog candidateではなくSelection Shortlistだけが分類されること
        - scene catalogもSelection Shortlistから作成されること
    """
    # Arrange
    images = [
        create_analyzed_image(
            path=f"/tmp/frame-{index:04d}.jpg",
            combined_features=_feature(index),
        )
        for index in range(550)
    ]
    scene_analyzer = CountingSceneAnalyzer()
    selector = AnalyzedImageSelector(
        config=SelectionConfig(
            ollama=OllamaConfig(model="gemma4", max_workers=4),
        ),
        metric_calculator=MetricCalculator(AnalyzerConfig()),
        scene_analyzer=scene_analyzer,
    )

    # Act
    _selected, _rejected, stats = selector.select(
        analyzed_images=images,
        num=10,
    )

    # Assert
    assert len(scene_analyzer.classified_paths) == 500
    assert "/tmp/frame-0499.jpg" in scene_analyzer.classified_paths
    assert "/tmp/frame-0500.jpg" not in scene_analyzer.classified_paths
    assert len(scene_analyzer.representative_paths) == 24
    assert set(scene_analyzer.representative_paths).issubset(
        set(scene_analyzer.classified_paths)
    )
    assert stats.rejected_by_selection_shortlist == 50


def test_select_keeps_selection_shortlist_at_least_requested_count() -> None:
    """選択要求枚数以上のSelection Shortlistが分類されること.

    Arrange:
        - 2000件を超える選択枚数が要求される
        - 要求枚数より多いblog candidateがある
    Act:
        - AnalyzedImageSelectorで選定される
    Assert:
        - Selection Shortlistが要求枚数未満に制限されないこと
    """
    # Arrange
    images = [
        create_analyzed_image(
            path=f"/tmp/large-frame-{index:04d}.jpg",
            combined_features=_feature(0, dim=2),
        )
        for index in range(2600)
    ]
    scene_analyzer = CountingSceneAnalyzer()
    selector = AnalyzedImageSelector(
        config=SelectionConfig(
            ollama=OllamaConfig(model="gemma4", max_workers=4),
        ),
        metric_calculator=MetricCalculator(AnalyzerConfig()),
        scene_analyzer=scene_analyzer,
    )

    # Act
    selector.select(
        analyzed_images=images,
        num=2500,
    )

    # Assert
    assert len(scene_analyzer.classified_paths) >= 2500


def test_select_uses_reserve_candidates_when_large_shortlist_has_failure() -> None:
    """大きなSelection Shortlistで分類失敗時も要求枚数が選定されること.

    Arrange:
        - 2000件を超える選択枚数が要求される
        - 要求枚数より多いblog candidateがある
        - Selection Shortlist内の1枚がOllama分類に失敗する
    Act:
        - AnalyzedImageSelectorで選定される
    Assert:
        - reserve候補から補われ、要求枚数が選定されること
    """
    # Arrange
    failed_path = "/tmp/reserve-frame-0000.jpg"
    images = [
        create_analyzed_image(
            path=f"/tmp/reserve-frame-{index:04d}.jpg",
            combined_features=_feature(index, dim=2600),
        )
        for index in range(2600)
    ]
    scene_analyzer = CountingSceneAnalyzer(failed_paths={failed_path})
    selector = AnalyzedImageSelector(
        config=SelectionConfig(
            ollama=OllamaConfig(model="gemma4", max_workers=4),
        ),
        metric_calculator=MetricCalculator(AnalyzerConfig()),
        scene_analyzer=scene_analyzer,
    )

    # Act
    selected, _rejected, stats = selector.select(
        analyzed_images=images,
        num=2500,
    )

    # Assert
    assert len(scene_analyzer.classified_paths) > 2500
    assert len(selected) == 2500
    assert stats.selected_count == 2500
    assert stats.ollama_classification_failed == 1


def test_select_refills_classification_after_shortlist_failures() -> None:
    """分類失敗が多い場合に未分類候補から補充されること.

    Arrange:
        - Selection Shortlistの通常上限に近い選択枚数が要求される
        - 初期Selection Shortlist内で多数のOllama分類失敗が発生する
        - 未分類のblog candidateが残っている
    Act:
        - AnalyzedImageSelectorで選定される
    Assert:
        - 未分類候補が追加分類され、要求枚数が選定されること
    """
    # Arrange
    failed_paths = {f"/tmp/refill-frame-{index:04d}.jpg" for index in range(101)}
    images = [
        create_analyzed_image(
            path=f"/tmp/refill-frame-{index:04d}.jpg",
            combined_features=_feature(index, dim=2105),
        )
        for index in range(2105)
    ]
    scene_analyzer = CountingSceneAnalyzer(failed_paths=failed_paths)
    selector = AnalyzedImageSelector(
        config=SelectionConfig(
            ollama=OllamaConfig(model="gemma4", max_workers=4),
        ),
        metric_calculator=MetricCalculator(AnalyzerConfig()),
        scene_analyzer=scene_analyzer,
    )

    # Act
    selected, _rejected, stats = selector.select(
        analyzed_images=images,
        num=1900,
    )

    # Assert
    assert len(scene_analyzer.classified_paths) > 2000
    assert len(selected) == 1900
    assert stats.selected_count == 1900
    assert stats.ollama_classification_failed == 101


def test_select_builds_shortlist_with_final_similarity_threshold() -> None:
    """最終選定の類似度しきい値でSelection Shortlistが多様化されること.

    Arrange:
        - 上位500件が最終選定では近すぎる画像である
        - 501件目に多様な画像がある
    Act:
        - AnalyzedImageSelectorで選定される
    Assert:
        - 低順位でも多様な画像がOllama分類へ進められること
    """
    # Arrange
    feature_dim = 502
    common_value = float(np.sqrt(0.9))
    unique_value = float(np.sqrt(0.1))
    images = [
        create_analyzed_image(
            path=f"/tmp/diverse-frame-{index:04d}.jpg",
            combined_features=_similar_feature(
                common_index=0,
                unique_index=index + 1,
                dim=feature_dim,
                common_value=common_value,
                unique_value=unique_value,
            ),
        )
        for index in range(500)
    ]
    images.append(
        create_analyzed_image(
            path="/tmp/diverse-frame-0500.jpg",
            combined_features=_feature(501, dim=feature_dim),
        )
    )
    scene_analyzer = CountingSceneAnalyzer()
    selector = AnalyzedImageSelector(
        config=SelectionConfig(
            ollama=OllamaConfig(model="gemma4", max_workers=4),
        ),
        metric_calculator=MetricCalculator(AnalyzerConfig()),
        scene_analyzer=scene_analyzer,
    )

    # Act
    selector.select(
        analyzed_images=images,
        num=2,
    )

    # Assert
    assert "/tmp/diverse-frame-0500.jpg" in scene_analyzer.classified_paths


def test_select_includes_frequent_patterns_in_scene_catalog_representatives() -> None:
    """頻出する見た目のpatternがscene catalog代表画像に含まれること.

    Arrange:
        - blur scoreが高い単発画像が24枚ある
        - blur scoreは低いが互いに似た頻出pattern画像が5枚ある
    Act:
        - AnalyzedImageSelectorで選定される
    Assert:
        - 代表画像24枚の中に頻出pattern画像が含まれること
    """
    # Arrange
    high_quality_images = [
        create_analyzed_image(
            path=f"/tmp/high-quality-{index:04d}.jpg",
            raw_metrics_dict={"blur_score": 200.0 - index},
            combined_features=_feature(index, dim=80),
        )
        for index in range(24)
    ]
    frequent_images = [
        create_analyzed_image(
            path=f"/tmp/frequent-pattern-{index:04d}.jpg",
            raw_metrics_dict={"blur_score": 80.0 - index},
            combined_features=_similar_feature(
                common_index=70,
                unique_index=71 + index,
                dim=80,
                common_value=float(np.sqrt(0.9)),
                unique_value=float(np.sqrt(0.1)),
            ),
        )
        for index in range(5)
    ]
    scene_analyzer = CountingSceneAnalyzer()
    selector = AnalyzedImageSelector(
        config=SelectionConfig(
            ollama=OllamaConfig(model="gemma4", max_workers=4),
        ),
        metric_calculator=MetricCalculator(AnalyzerConfig()),
        scene_analyzer=scene_analyzer,
    )

    # Act
    selector.select(
        analyzed_images=high_quality_images + frequent_images,
        num=10,
    )

    # Assert
    assert len(scene_analyzer.representative_paths) == 24
    assert any(
        path.startswith("/tmp/frequent-pattern-")
        for path in scene_analyzer.representative_paths
    )


def test_select_uses_fallback_scene_when_catalog_generation_fails() -> None:
    """catalog作成失敗時にfallback sceneで候補が選定されること.

    Arrange:
        - 選定対象画像とcatalog作成に失敗するscene analyzerがある
    Act:
        - AnalyzedImageSelectorで選定される
    Assert:
        - 例外を送出せずfallback sceneで候補が選定されること
        - catalog fallbackの発生理由が統計に記録されること
    """
    # Arrange
    first = create_analyzed_image(
        path="/tmp/first.jpg",
        combined_features=_feature(1),
    )
    second = create_analyzed_image(
        path="/tmp/second.jpg",
        combined_features=_feature(100),
    )
    selector = AnalyzedImageSelector(
        config=SelectionConfig(
            ollama=OllamaConfig(model="gemma4"),
        ),
        metric_calculator=MetricCalculator(AnalyzerConfig()),
        scene_analyzer=FakeSceneAnalyzer(catalog_error=OSError("timed out")),
    )

    # Act
    selected, rejected, stats = selector.select(
        analyzed_images=[first, second],
        num=2,
    )

    # Assert
    assert {candidate.path for candidate in selected} == {
        "/tmp/first.jpg",
        "/tmp/second.jpg",
    }
    assert rejected == []
    assert stats.selected_count == 2
    assert stats.ollama_classification_failed == 0
    assert stats.ollama_classification_failure_rate == 0.0
    assert stats.ollama_catalog_fallback_used is True
    assert stats.ollama_catalog_fallback_reason == "OSError: timed out"
    assert stats.scene_catalog[0].slug == "fallback"
    assert stats.scene_distribution == {"fallback": 2}


class _FakeSceneAnalyzer:
    """テスト用のscene analyzer."""

    def __init__(
        self,
        classifications_by_path: dict[str, SceneClassification],
    ) -> None:
        """fake analyzerを初期化する."""
        self.classifications_by_path = classifications_by_path
        self.catalog = [
            SceneCatalogEntry(
                "battle",
                "戦闘",
                "敵と戦う場面",
                SceneSelectionRole.RECURRING_GAMEPLAY,
            ),
            SceneCatalogEntry(
                "conversation",
                "会話",
                "人物同士の会話場面",
                SceneSelectionRole.CINEMATIC,
            ),
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


class _ConcurrentSceneAnalyzer:
    """並列実行を観測するscene analyzer."""

    def __init__(self) -> None:
        """fake analyzerを初期化する."""
        self.catalog = [
            SceneCatalogEntry("battle", "戦闘", "敵と戦う場面"),
            SceneCatalogEntry("other", "その他", "分類しにくい場面"),
            SceneCatalogEntry("conversation", "会話", "人物同士の会話場面"),
        ]
        self._lock = threading.Lock()
        self._active = 0
        self.max_active = 0

    def generate_scene_catalog(
        self,
        representative_paths: list[str],
        scene_hint: str | None,
    ) -> list[SceneCatalogEntry]:
        """scene catalogを返す."""
        assert representative_paths
        assert scene_hint is None
        return self.catalog

    def classify_image(
        self,
        image_path: str,
        catalog: list[SceneCatalogEntry],
    ) -> SceneClassification | None:
        """同時実行数を記録して分類結果を返す."""
        assert catalog == self.catalog
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            time.sleep(0.05)
            return SceneClassification(
                scene_slug="battle",
                scene_display_name="戦闘",
                scene_description=f"{image_path}の戦闘場面",
                confidence=0.9,
            )
        finally:
            with self._lock:
                self._active -= 1
