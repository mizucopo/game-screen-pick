"""output_planner.py の単体テスト."""

from pathlib import Path

import pytest

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
        scene_slug=scene_label,
        scene_display_name=scene_label,
        scene_description=scene_label,
        scene_confidence=0.5,
        quality_score=0.6,
        selection_score=0.6,
        score_band="high",
        variant_group=f"{scene_label}_001",
        outlier_rejected=False,
    )


def _build_output_record(
    source_paths: list[str],
    scene_labels: list[str],
) -> OutputRecord:
    play_count = scene_labels.count("play")
    event_count = scene_labels.count("event")
    scene_counts = {"play": play_count, "event": event_count}

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
        rejected_by_selection_shortlist=0,
        selected_count=len(source_paths),
        scene_distribution=dict(scene_counts),
        scene_mix_target=dict(scene_counts),
        scene_mix_actual=dict(scene_counts),
        threshold_relaxation_steps=[0.72],
        content_filter_breakdown={},
        whole_input_profile=None,
        scene_catalog=[],
        ollama_catalog_fallback_used=False,
        ollama_catalog_fallback_reason=None,
        ollama_classification_failed=0,
        ollama_classification_failure_rate=0.0,
    )


@pytest.mark.parametrize(
    "scene_name,index,suffix,requested_num,expected",
    [
        ("play", 1, ".jpg", 3, "play0001.jpg"),
        ("event", 12, ".png", 9999, "event0012.png"),
        ("play", 1, ".bmp", 10000, "play00001.bmp"),
    ],
)
def test_build_scene_numbered_filename_uses_scene_prefix_and_padding(
    scene_name: str,
    index: int,
    suffix: str,
    requested_num: int,
    expected: str,
) -> None:
    """scene名と要求枚数に応じたファイル名が生成されること.

    Arrange:
        - scene名、連番index、拡張子、要求枚数がある
    Act:
        - build_scene_numbered_filenameが実行される
    Assert:
        - scene名 + ゼロ埋め連番 + 拡張子の形式で返されること
        - 要求枚数が4桁以下なら4桁、それ以上なら要求枚数の桁数でゼロ埋めされること
    """
    # Arrange - パラメタライズド引数からscene名、連番、拡張子、要求枚数を設定
    # Act
    result = OutputPlanner.build_scene_numbered_filename(
        scene_name=scene_name,
        index=index,
        suffix=suffix,
        requested_num=requested_num,
    )

    # Assert
    assert result == expected


def test_output_planner_plans_scene_numbered_paths_without_copying_files(
    tmp_path: Path,
) -> None:
    """コピーなしでscene別連番と衝突回避済み出力パスが計画されること.

    Arrange:
        - play 2件、event 1件のoutput recordがある
        - 出力先には既存のplay0001.jpgがあるものとして扱われる
    Act:
        - 出力計画が作成される
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


def test_output_planner_rejects_missing_requested_num(
    tmp_path: Path,
) -> None:
    """要求枚数なしではscene別連番の出力計画が作成されないこと.

    Arrange:
        - output recordがある
    Act:
        - requested_numなしで出力計画が作成される
    Assert:
        - ValueErrorが送出されること
    """
    # Arrange
    source_path = str(tmp_path / "source" / "screen.png")
    record = _build_output_record([source_path], ["play"])
    output_dir = tmp_path / "output"

    # Act / Assert
    with pytest.raises(ValueError, match="requested_num"):
        OutputPlanner.plan_selected_outputs(
            record,
            str(output_dir),
            requested_num=None,
        )


def test_output_planner_avoids_case_insensitive_filename_collision(
    tmp_path: Path,
) -> None:
    """大文字小文字だけが異なる既存ファイルとの衝突が回避されること.

    Arrange:
        - 出力先には既存のScreen.pngがあるものとして扱われる
        - 選択候補のscene slugはplayである
    Act:
        - 出力計画が作成される
    Assert:
        - 大文字小文字を区別しないfilesystemで上書きされない出力パスに計画されること
        - filesystem copyが実行されないこと
    """
    # Arrange
    source_path = str(tmp_path / "source" / "screen.png")
    record = _build_output_record([source_path], ["play"])
    output_dir = tmp_path / "output"

    # Act
    result = OutputPlanner.plan_selected_outputs(
        record,
        str(output_dir),
        requested_num=1,
        existing_filenames=["Play0001.png"],
    )

    # Assert
    assert result.selected[0].output_path == str(
        (output_dir / "play0001_1.png").resolve()
    )
    assert not output_dir.exists()
