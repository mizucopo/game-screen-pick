"""output_record.py の単体テスト."""

from src.constants.scene_label import SceneLabel
from src.models.output_record import OutputRecord
from src.models.picker_statistics import PickerStatistics
from src.models.selection_annotation import SelectionAnnotation
from tests.conftest import create_scored_candidate


def test_output_record_projects_selection_for_output_adapters() -> None:
    """出力adapterが利用する安定したrecordへ射影されること.

    Arrange:
        - 選択候補と除外候補にscene scoreが設定されている
        - 統計情報に集計値と選定注釈が設定されている
    Act:
        - OutputRecordへ射影される
        - コピー後の出力パスが反映される
    Assert:
        - 候補と統計情報の出力に必要な値がrecordから取得されること
        - 出力パス付きのrecordが生成されること
    """
    # Arrange
    selected = [
        create_scored_candidate(
            path="/tmp/play.jpg",
            scene_label=SceneLabel.PLAY,
            play_score=0.81234,
            event_score=0.12345,
            density_score=0.73456,
            selection_score=0.65432,
        )
    ]
    rejected = [
        create_scored_candidate(
            path="/tmp/event.jpg",
            scene_label=SceneLabel.EVENT,
            play_score=0.23456,
            event_score=0.87654,
            density_score=0.12345,
            selection_score=0.54321,
        )
    ]
    stats = PickerStatistics(
        total_files=2,
        analyzed_ok=2,
        analyzed_fail=0,
        rejected_by_similarity=1,
        rejected_by_content_filter=0,
        selected_count=1,
        resolved_profile="active",
        scene_distribution={"play": 1, "event": 1},
        scene_mix_target={"play": 1, "event": 0},
        scene_mix_actual={"play": 1, "event": 0},
        threshold_relaxation_steps=[0.72],
        content_filter_breakdown={"blackout": 0},
        whole_input_profile=None,
        selection_annotations_by_path={
            "/tmp/play.jpg": SelectionAnnotation(score_band="high"),
            "/tmp/event.jpg": SelectionAnnotation(
                score_band="outlier",
                outlier_rejected=True,
            ),
        },
    )

    # Act
    record = OutputRecord.from_selection(selected, rejected, stats)
    record_with_paths = record.with_selected_output_paths(
        {"/tmp/play.jpg": "/tmp/output/play0001.jpg"}
    )

    # Assert
    assert record.resolved_profile == "active"
    assert record.scene_distribution == {"play": 1, "event": 1}
    assert record.total_files == 2
    assert record.selected[0].source_path == "/tmp/play.jpg"
    assert record.selected[0].filename == "play.jpg"
    assert record.selected[0].scene_label == "play"
    assert record.selected[0].play_score == 0.8123
    assert record.selected[0].score_band == "high"
    assert record.rejected[0].scene_label == "event"
    assert record.rejected[0].outlier_rejected is True
    assert record_with_paths.selected[0].output_path == "/tmp/output/play0001.jpg"
    assert record.selected[0].output_path is None
