"""file_utils.pyの単体テスト."""

from pathlib import Path

import pytest

from src.models.output_candidate_record import OutputCandidateRecord
from src.models.output_record import OutputRecord
from src.utils.file_utils import FileUtils


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
    scene_labels: list[str] | None = None,
) -> OutputRecord:
    scene_labels = scene_labels or ["play"] * len(source_paths)
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
        scene_distribution={"play": len(source_paths), "event": 0},
        scene_mix_target={"play": len(source_paths), "event": 0},
        scene_mix_actual={"play": len(source_paths), "event": 0},
        threshold_relaxation_steps=[0.72],
        content_filter_breakdown={},
        whole_input_profile=None,
        scene_catalog=[],
        ollama_catalog_fallback_used=False,
        ollama_catalog_fallback_reason=None,
        ollama_classification_failed=0,
        ollama_classification_failure_rate=0.0,
    )


def test_copy_selected_items_returns_output_record_with_copied_paths(
    tmp_path: Path,
) -> None:
    """output recordの選択候補がコピーされ出力パスが反映されること.

    Arrange:
        - ソース画像ファイルを持つoutput recordがある
        - コピー先ディレクトリを用意する
    Act:
        - copy_selected_itemsが実行される
    Assert:
        - ファイルがコピーされること
        - 戻り値のoutput recordにコピー先パスが設定されること
    """
    # Arrange
    src_file = tmp_path / "source" / "test.png"
    src_file.parent.mkdir()
    src_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    dest_dir = tmp_path / "output"

    # Act
    result = FileUtils.copy_selected_items(
        _build_output_record([str(src_file)]),
        str(dest_dir),
        requested_num=1,
    )

    # Assert
    assert (dest_dir / "play0001.png").exists()
    assert result.selected[0].output_path == str((dest_dir / "play0001.png").resolve())


def test_copy_planned_outputs_creates_output_parent_directories(
    tmp_path: Path,
) -> None:
    """計画済みcopy adapterが出力先親ディレクトリを作成すること.

    Arrange:
        - ソース画像ファイルを持つoutput recordがある
        - 存在しない出力ディレクトリへのoutput_pathが計画されている
    Act:
        - copy_planned_outputsが実行される
    Assert:
        - 出力先親ディレクトリが作成されること
        - 計画済みの出力パスへファイルがコピーされること
    """
    # Arrange
    src_file = tmp_path / "source" / "test.png"
    src_file.parent.mkdir()
    src_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    dest_dir = tmp_path / "missing" / "output"
    output_record = _build_output_record([str(src_file)]).with_selected_output_paths(
        {str(src_file): str((dest_dir / "test.png").resolve())}
    )

    # Act
    FileUtils.copy_planned_outputs(output_record)

    # Assert
    assert (dest_dir / "test.png").exists()


def test_copy_selected_items_rejects_non_empty_output_dir(
    tmp_path: Path,
) -> None:
    """出力ディレクトリが空でない場合はコピーされないこと.

    Arrange:
        - 出力ディレクトリに既存ファイル（play0001.jpg）がある
        - 選択画像がある
    Act:
        - copy_selected_itemsを実行する
    Assert:
        - ValueErrorが送出されること
        - 既存ファイル以外はコピーされないこと
    """
    # Arrange
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "play0001.jpg").write_bytes(b"existing")

    gameplay1 = tmp_path / "play1.jpg"
    gameplay1.write_bytes(b"fake_image_data")

    record = _build_output_record([str(gameplay1)], ["play"])

    # Act & Assert
    with pytest.raises(ValueError, match="出力フォルダは空である必要があります"):
        FileUtils.copy_selected_items(
            record,
            str(output_dir),
            requested_num=1,
        )
    assert sorted(path.name for path in output_dir.iterdir()) == ["play0001.jpg"]


def test_copy_selected_items_copies_files(tmp_path: Path) -> None:
    """選択されたアイテムがコピー先にコピーされること.

    Arrange:
        - ソース画像ファイルを作成する
        - コピー先ディレクトリを用意する
        - output recordを作成する
    Act:
        - copy_selected_itemsを呼び出す
    Assert:
        - 戻り値の選択候補が1件であること
        - コピー先にファイルが存在すること
    """
    # Arrange
    src_file = tmp_path / "source" / "test.png"
    src_file.parent.mkdir()
    src_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    dest_dir = tmp_path / "output"

    # Act
    result = FileUtils.copy_selected_items(
        _build_output_record([str(src_file)]),
        str(dest_dir),
        requested_num=1,
    )

    # Assert
    assert len(result.selected) == 1
    assert (dest_dir / "play0001.png").exists()


def test_copy_selected_items_writes_scene_numbered_names(tmp_path: Path) -> None:
    """scene別連番ファイル名が付けられること.

    Arrange:
        - 異なる拡張子の画像ファイルを2件作成する
        - コピー先ディレクトリを用意する
        - output recordを作成する
    Act:
        - requested_num=2でcopy_selected_itemsを呼び出す
    Assert:
        - 戻り値が2件であること
        - コピー先にplay0001.png, play0002.jpgが生成されること
    """
    # Arrange
    for name in ["a.png", "b.jpg"]:
        src_file = tmp_path / "source" / name
        src_file.parent.mkdir(parents=True, exist_ok=True)
        src_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    dest_dir = tmp_path / "output"
    record = _build_output_record(
        [
            str(tmp_path / "source" / "a.png"),
            str(tmp_path / "source" / "b.jpg"),
        ]
    )

    # Act
    result = FileUtils.copy_selected_items(
        record,
        str(dest_dir),
        requested_num=2,
    )

    # Assert
    assert len(result.selected) == 2
    files = sorted(dest_dir.iterdir())
    assert files[0].name == "play0001.png"
    assert files[1].name == "play0002.jpg"


def test_copy_selected_items_raises_without_requested_num(
    tmp_path: Path,
) -> None:
    """requested_num=Noneの場合、ValueErrorが送出されること.

    Arrange:
        - ソース画像ファイルを作成する
        - コピー先ディレクトリを用意する
        - output recordを作成する
    Act:
        - requested_num指定なしでcopy_selected_itemsを呼び出す
    Assert:
        - ValueErrorが送出されること
    """
    # Arrange
    src_file = tmp_path / "source" / "test.png"
    src_file.parent.mkdir()
    src_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    dest_dir = tmp_path / "output"

    # Act & Assert
    with pytest.raises(ValueError, match="requested_num"):
        FileUtils.copy_selected_items(
            _build_output_record([str(src_file)]),
            str(dest_dir),
            requested_num=None,
        )


def test_copy_selected_items_numbers_duplicate_scene_outputs(tmp_path: Path) -> None:
    """同じsceneの画像に連番ファイル名が生成されること.

    Arrange:
        - 同じsceneに属する画像ファイルを2件作成する
        - コピー先ディレクトリを用意する
        - output recordを作成する
    Act:
        - copy_selected_itemsを呼び出す
    Assert:
        - 戻り値が2件であること
        - コピー先に2ファイルが存在すること
        - play0001.pngとplay0002.pngが生成されること
    """
    # Arrange
    for index, name in enumerate(["test.png", "test.png"]):
        src_dir = tmp_path / f"source{index}"
        src_dir.mkdir()
        src_file = src_dir / name
        src_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    dest_dir = tmp_path / "output"
    record = _build_output_record(
        [
            str(tmp_path / "source0" / "test.png"),
            str(tmp_path / "source1" / "test.png"),
        ]
    )

    # Act
    result = FileUtils.copy_selected_items(record, str(dest_dir), requested_num=2)

    # Assert
    assert len(result.selected) == 2
    copied_files = list(dest_dir.iterdir())
    assert len(copied_files) == 2
    names = {file.name for file in copied_files}
    assert "play0001.png" in names
    assert "play0002.png" in names
