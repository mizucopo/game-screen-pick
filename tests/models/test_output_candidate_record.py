"""output_candidate_record.py の単体テスト."""

from src.models.output_candidate_record import OutputCandidateRecord
from src.models.selection_annotation import SelectionAnnotation
from tests.conftest import create_scored_candidate


def test_output_candidate_record_projects_scored_candidate() -> None:
    """候補内部構造が出力adapter向けrecordへ射影されること.

    Arrange:
        - scene scoreと選定注釈を持つ候補がある
    Act:
        - OutputCandidateRecordへ射影される
    Assert:
        - 出力adapterに必要な値だけが丸め済みで保持されること
    """
    # Arrange
    candidate = create_scored_candidate(
        path="/tmp/battle.jpg",
        scene_slug="battle",
        scene_display_name="戦闘",
        scene_description="敵との戦闘場面",
        scene_confidence=0.56789,
        quality_score=0.67891,
        selection_score=0.65432,
    )
    annotation = SelectionAnnotation(
        score_band="high",
        outlier_rejected=True,
        variant_group="battle_001",
    )

    # Act
    record = OutputCandidateRecord.from_scored_candidate(candidate, annotation)

    # Assert
    assert record.source_path == "/tmp/battle.jpg"
    assert record.filename == "battle.jpg"
    assert record.suffix == ".jpg"
    assert record.scene_slug == "battle"
    assert record.scene_display_name == "戦闘"
    assert record.scene_description == "敵との戦闘場面"
    assert record.scene_confidence == 0.5679
    assert record.quality_score == 0.6789
    assert record.selection_score == 0.6543
    assert record.score_band == "high"
    assert record.variant_group == "battle_001"
    assert record.outlier_rejected is True
