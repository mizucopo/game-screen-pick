"""AnalyzedImageSelector の単体テスト."""

from src.analyzers.metric_calculator import MetricCalculator
from src.models.analyzer_config import AnalyzerConfig
from src.models.scene_mix import SceneMix
from src.models.selection_config import SelectionConfig
from src.services.analyzed_image_selector import AnalyzedImageSelector
from tests.conftest import _feature, create_analyzed_image


def test_select_returns_filtered_scene_mix_and_statistics() -> None:
    """解析済み画像から選定結果と統計が生成されること.

    Arrange:
        - content filter で除外される画像と選定対象の画像がある
        - play/event を 1 枚ずつ選ぶ scene mix が指定されている
    Act:
        - AnalyzedImageSelectorで選定される
    Assert:
        - 除外画像を含まない選定結果が返されること
        - 選定統計に content filter と scene mix の結果が反映されること
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
    play = create_analyzed_image(
        path="/tmp/play.jpg",
        combined_features=_feature(1),
    )
    event = create_analyzed_image(
        path="/tmp/event.jpg",
        combined_features=_feature(100),
    )
    selector = AnalyzedImageSelector(
        config=SelectionConfig(
            profile="active",
            scene_mix=SceneMix(play=0.5, event=0.5),
        ),
        metric_calculator=MetricCalculator(AnalyzerConfig()),
    )

    # Act
    selected, rejected, stats = selector.select(
        analyzed_images=[dark, play, event],
        num=2,
        total_files=4,
        analyzed_fail=1,
    )

    # Assert
    assert {candidate.path for candidate in selected} == {
        "/tmp/play.jpg",
        "/tmp/event.jpg",
    }
    assert rejected == []
    assert stats.total_files == 4
    assert stats.analyzed_ok == 3
    assert stats.analyzed_fail == 1
    assert stats.selected_count == 2
    assert stats.rejected_by_content_filter == 1
    assert stats.content_filter_breakdown["blackout"] == 1
    assert stats.scene_mix_target == {"play": 1, "event": 1}
    assert stats.scene_mix_actual == {"play": 1, "event": 1}
    assert stats.resolved_profile == "active"
