"""output_planner.py の単体テスト."""

from pathlib import Path

from src.models.output_candidate_record import OutputCandidateRecord
from src.models.output_record import OutputRecord
from src.services.output_planner import OutputPlanner


def _build_candidate(
    source_path: str,
    scene_label: str = "play",
) -> OutputCandidateRecord:
    path = Path(source_path)
    return OutputCandidateRecord(
        source_path=source_path,
        filename=path.name,
        suffix=path.suffix,
        scene_label=scene_label,
        play_score=0.8,
        event_score=0.3,
        density_score=0.7,
        scene_confidence=0.5,
        quality_score=0.6,
        selection_score=0.6,
        score_band="high",
        outlier_rejected=False,
    )


def _build_output_record(
    source_paths: list[str],
    scene_labels: list[str],
) -> OutputRecord:
    return OutputRecord(
        selected=[
            _build_candidate(source_path, scene_label)
            for source_path, scene_label in zip(source_paths, scene_labels, strict=True)
        ],
        rejected=[],
        total_files=len(source_paths),
        analyzed_ok=len(source_paths),
        analyzed_fail=0,
        rejected_by_similarity=0,
        rejected_by_content_filter=0,
        selected_count=len(source_paths),
        resolved_profile="active",
        scene_distribution={"play": 2, "event": 1},
        scene_mix_target={"play": 2, "event": 1},
        scene_mix_actual={"play": 2, "event": 1},
        threshold_relaxation_steps=[0.72],
        content_filter_breakdown={},
        whole_input_profile=None,
    )


def test_output_planner_plans_renamed_paths_without_copying_files(
    tmp_path: Path,
) -> None:
    """コピーなしでscene別連番と衝突回避済み出力パスが計画されること.

    Arrange:
        - play 2件、event 1件のoutput recordがある
        - 出力先には既存のplay0001.jpgがあるものとして扱われる
    Act:
        - rename=Trueで出力計画が作成される
    Assert:
        - scene別連番の出力パスが選択候補に反映されること
        - 既存ファイル名との衝突が回避されること
        - filesystem copyが実行されないこと
    """
    # Arrange
    source_paths = [
        str(tmp_path / "play1.jpg"),
        str(tmp_path / "event1.jpg"),
        str(tmp_path / "play2.jpg"),
    ]
    record = _build_output_record(source_paths, ["play", "event", "play"])
    output_dir = tmp_path / "output"

    # Act
    result = OutputPlanner.plan_selected_outputs(
        record,
        str(output_dir),
        rename=True,
        requested_num=3,
        existing_filenames=["play0001.jpg"],
    )

    # Assert
    assert result.selected[0].output_path == str(
        (output_dir / "play0001_1.jpg").resolve()
    )
    assert result.selected[1].output_path == str(
        (output_dir / "event0001.jpg").resolve()
    )
    assert result.selected[2].output_path == str(
        (output_dir / "play0002.jpg").resolve()
    )
    assert not output_dir.exists()


def test_output_planner_plans_duplicate_filenames_without_rename(
    tmp_path: Path,
) -> None:
    """通常出力で同名ファイルが衝突しない出力パスに計画されること.

    Arrange:
        - 異なるsource_pathに同じfilenameを持つoutput recordがある
    Act:
        - rename=Falseで出力計画が作成される
    Assert:
        - 1件目は元のファイル名で計画されること
        - 2件目はサフィックス付きのファイル名で計画されること
        - filesystem copyが実行されないこと
    """
    # Arrange
    source_paths = [
        str(tmp_path / "source1" / "screen.png"),
        str(tmp_path / "source2" / "screen.png"),
    ]
    record = _build_output_record(source_paths, ["play", "event"])
    output_dir = tmp_path / "output"

    # Act
    result = OutputPlanner.plan_selected_outputs(
        record,
        str(output_dir),
    )

    # Assert
    assert result.selected[0].output_path == str((output_dir / "screen.png").resolve())
    assert result.selected[1].output_path == str(
        (output_dir / "screen_1.png").resolve()
    )
    assert not output_dir.exists()
