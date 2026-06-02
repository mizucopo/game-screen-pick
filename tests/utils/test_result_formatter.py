"""ResultFormatter のテスト."""

from io import StringIO
from unittest.mock import patch

from src.models.picker_statistics import PickerStatistics
from src.models.selection_annotation import SelectionAnnotation
from src.utils.result_formatter import ResultFormatter
from tests.conftest import create_scored_candidate


def test_display_results_runs_without_error() -> None:
    """display_results がエラーなく実行されること.

    Arrange:
        - 候補画像と統計情報を用意する
    Act:
        - display_results を呼び出す
    Assert:
        - 例外が発生せずに完了されること
    """
    # Arrange
    candidate = create_scored_candidate(path="/tmp/test_image.jpg")

    stats = PickerStatistics(
        total_files=1,
        analyzed_ok=1,
        analyzed_fail=0,
        rejected_by_similarity=0,
        rejected_by_content_filter=0,
        selected_count=1,
        resolved_profile="active",
        scene_distribution={"play": 1, "event": 0},
        scene_mix_target={"play": 1, "event": 0},
        scene_mix_actual={"play": 1, "event": 0},
        threshold_relaxation_steps=[0.7],
        content_filter_breakdown={},
        whole_input_profile=None,
    )

    # Act
    with patch("sys.stdout", new_callable=StringIO):
        ResultFormatter.display_results([candidate], stats)

    # Assert — 例外なく完了すればOK


def test_display_results_uses_selection_annotations_from_stats() -> None:
    """選定注釈のscore_bandが表示に使われること.

    Arrange:
        - 候補にはscore_bandが設定されていない
        - 統計情報に候補パスごとの選定注釈がある
    Act:
        - display_resultsが実行される
    Assert:
        - 表示ログに選定注釈のscore_bandが含まれること
    """
    # Arrange
    candidate = create_scored_candidate(path="/tmp/test_image.jpg")
    stats = PickerStatistics(
        total_files=1,
        analyzed_ok=1,
        analyzed_fail=0,
        rejected_by_similarity=0,
        rejected_by_content_filter=0,
        selected_count=1,
        resolved_profile="active",
        scene_distribution={"play": 1, "event": 0},
        scene_mix_target={"play": 1, "event": 0},
        scene_mix_actual={"play": 1, "event": 0},
        threshold_relaxation_steps=[0.7],
        content_filter_breakdown={},
        whole_input_profile=None,
        selection_annotations_by_path={
            candidate.path: SelectionAnnotation(score_band="high")
        },
    )

    # Act
    with patch("src.utils.result_formatter.logger") as logger:
        ResultFormatter.display_results([candidate], stats)

    # Assert
    messages = [call.args[0] for call in logger.info.call_args_list]
    assert any("band: high" in message for message in messages)
