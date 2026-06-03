"""result_formatter.py の単体テスト."""

from unittest.mock import patch

from src.models.output_candidate_record import OutputCandidateRecord
from src.models.output_record import OutputRecord
from src.utils.result_formatter import ResultFormatter


def _build_output_record() -> OutputRecord:
    return OutputRecord(
        selected=[
            OutputCandidateRecord(
                source_path="/tmp/test_image.jpg",
                filename="test_image.jpg",
                suffix=".jpg",
                scene_slug="battle",
                scene_display_name="戦闘",
                scene_description="敵との戦闘場面",
                scene_confidence=0.5,
                quality_score=0.6,
                selection_score=0.6,
                score_band="high",
                variant_group="battle_001",
                outlier_rejected=False,
            )
        ],
        rejected=[],
        total_files=1,
        analyzed_ok=1,
        analyzed_fail=0,
        rejected_by_similarity=0,
        rejected_by_content_filter=0,
        selected_count=1,
        scene_distribution={"battle": 1},
        scene_mix_target={"battle": 1},
        scene_mix_actual={"battle": 1},
        threshold_relaxation_steps=[0.7],
        content_filter_breakdown={},
        whole_input_profile=None,
        scene_catalog=[
            {
                "slug": "battle",
                "display_name": "戦闘",
                "description": "敵との戦闘場面",
            }
        ],
        ollama_classification_failed=1,
        ollama_classification_failure_rate=0.25,
    )


def test_display_results_runs_without_error() -> None:
    """display_resultsがエラーなく実行されること.

    Arrange:
        - 候補画像と統計情報を持つoutput recordを用意する
    Act:
        - display_resultsを呼び出す
    Assert:
        - 例外が発生せずに完了されること
    """
    # Arrange
    record = _build_output_record()

    # Act
    ResultFormatter.display_results(record)

    # Assert - 例外なく完了すればOK


def test_display_results_uses_output_record() -> None:
    """output recordから既存の表示内容が出力されること.

    Arrange:
        - 選択候補と統計値を持つoutput recordがある
    Act:
        - display_resultsが実行される
    Assert:
        - 候補情報と統計情報がログへ出力されること
    """
    # Arrange
    record = _build_output_record()

    # Act
    with patch("src.utils.result_formatter.logger") as logger:
        ResultFormatter.display_results(record)

    # Assert
    messages = [call.args[0] for call in logger.info.call_args_list]
    assert any("[1] test_image.jpg" in message for message in messages)
    assert any("戦闘" in message for message in messages)
    assert any("band: high" in message for message in messages)
    assert "総ファイル数: 1" in messages
    assert "解析成功: 1" in messages
    assert "Ollama分類失敗: 1" in messages
    assert "Ollama分類失敗率: 25.00%" in messages
    assert not any("プロファイル:" in message for message in messages)
