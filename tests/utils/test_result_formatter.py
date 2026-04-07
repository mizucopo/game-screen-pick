"""ResultFormatter のテスト."""

from io import StringIO
from unittest.mock import patch

from src.models.picker_statistics import PickerStatistics
from src.utils.result_formatter import ResultFormatter
from tests.conftest import create_scored_candidate


def test_display_results_runs_without_error() -> None:
    """display_results がエラーなく実行されること."""
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

    # 標準出力をキャプチャしてエラーなく完了することを確認
    with patch("sys.stdout", new_callable=StringIO):
        ResultFormatter.display_results([candidate], stats)
