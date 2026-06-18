"""output_record.py の単体テスト."""

from src.models.output_record import OutputRecord
from src.models.picker_statistics import PickerStatistics
from src.models.scene_catalog_entry import SceneCatalogEntry
from src.models.scene_selection_role import SceneSelectionRole
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
            path="/tmp/battle.jpg",
            scene_slug="battle",
            scene_display_name="戦闘",
            scene_description="敵との戦闘場面",
            scene_selection_role=SceneSelectionRole.RECURRING_GAMEPLAY,
            selection_score=0.65432,
        )
    ]
    rejected = [
        create_scored_candidate(
            path="/tmp/conversation.jpg",
            scene_slug="conversation",
            scene_display_name="会話",
            scene_description="人物同士の会話場面",
            scene_selection_role=SceneSelectionRole.CINEMATIC,
            selection_score=0.54321,
        )
    ]
    stats = PickerStatistics(
        total_files=2,
        analyzed_ok=2,
        analyzed_fail=0,
        rejected_by_similarity=1,
        rejected_by_content_filter=0,
        rejected_by_selection_shortlist=2,
        selected_count=1,
        scene_distribution={"battle": 1, "conversation": 1},
        scene_mix_target={"battle": 1, "conversation": 0},
        scene_mix_actual={"battle": 1, "conversation": 0},
        threshold_relaxation_steps=[0.72],
        content_filter_breakdown={"blackout": 0},
        whole_input_profile=None,
        selection_annotations_by_path={
            "/tmp/battle.jpg": SelectionAnnotation(
                score_band="high",
                variant_group="battle_001",
            ),
            "/tmp/conversation.jpg": SelectionAnnotation(
                score_band="outlier",
                outlier_rejected=True,
                variant_group="conversation_001",
            ),
        },
        scene_catalog=[
            SceneCatalogEntry(
                "battle",
                "戦闘",
                "敵との戦闘場面",
                SceneSelectionRole.RECURRING_GAMEPLAY,
            ),
            SceneCatalogEntry(
                "conversation",
                "会話",
                "人物同士の会話場面",
                SceneSelectionRole.CINEMATIC,
            ),
            SceneCatalogEntry("other", "その他", "分類しにくい場面"),
        ],
        ollama_classification_failed=1,
        ollama_classification_failure_rate=0.25,
    )

    # Act
    record = OutputRecord.from_selection(selected, rejected, stats)
    record_with_paths = record.with_selected_output_paths(
        {"/tmp/battle.jpg": "/tmp/output/battle0001.jpg"}
    )

    # Assert
    assert record.scene_distribution == {"battle": 1, "conversation": 1}
    assert record.scene_catalog[0]["slug"] == "battle"
    assert record.scene_catalog[0]["selection_role"] == "recurring_gameplay"
    assert record.ollama_classification_failed == 1
    assert record.ollama_classification_failure_rate == 0.25
    assert record.ollama_catalog_fallback_used is False
    assert record.ollama_catalog_fallback_reason is None
    assert record.total_files == 2
    assert record.rejected_by_selection_shortlist == 2
    assert record.selected[0].source_path == "/tmp/battle.jpg"
    assert record.selected[0].filename == "battle.jpg"
    assert record.selected[0].scene_slug == "battle"
    assert record.selected[0].scene_display_name == "戦闘"
    assert record.selected[0].scene_selection_role == "recurring_gameplay"
    assert record.selected[0].variant_group == "battle_001"
    assert record.selected[0].score_band == "high"
    assert record.rejected[0].scene_slug == "conversation"
    assert record.rejected[0].scene_selection_role == "cinematic"
    assert record.rejected[0].outlier_rejected is True
    assert record_with_paths.selected[0].output_path == "/tmp/output/battle0001.jpg"
    assert record.selected[0].output_path is None
