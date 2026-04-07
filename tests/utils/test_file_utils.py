"""file_utils.pyの単体テスト."""

from pathlib import Path

import pytest

from src.constants.scene_label import SceneLabel
from src.utils.file_utils import FileUtils
from tests.conftest import create_scored_candidate


@pytest.mark.parametrize(
    "existing_files,filename,expected",
    [
        ([], "image.jpg", "image.jpg"),
        (["image.jpg"], "image.jpg", "image_1.jpg"),
        # 飛び番号がある場合の最小値取得
        (["image.jpg", "image_1.jpg", "image_3.jpg"], "image.jpg", "image_2.jpg"),
        # 複数拡張子と特殊文字
        (["archive.tar.gz"], "archive.tar.gz", "archive.tar_1.gz"),
        (["画像ファイル.jpg"], "画像ファイル.jpg", "画像ファイル_1.jpg"),
    ],
)
def test_get_unique_destination_generates_unique_filename(
    tmp_path: Path,
    existing_files: list[str],
    filename: str,
    expected: str,
) -> None:
    """既存ファイルに基づいて一意なファイル名が生成されること.

    Arrange:
        - 出力ディレクトリに既存ファイルが存在する（または空である）
    Act:
        - get_unique_destinationが実行される
    Assert:
        - 適切なサフィックス付きのファイル名が返されること
    """
    # Arrange
    for f in existing_files:
        (tmp_path / f).touch()

    # Act
    result = FileUtils.get_unique_destination(tmp_path, filename)

    # Assert
    assert result == tmp_path / expected


@pytest.mark.parametrize(
    "scene_name,index,suffix,requested_num,expected",
    [
        ("play", 1, ".jpg", 3, "play0001.jpg"),
        ("event", 12, ".png", 9999, "event0012.png"),
        ("play", 1, ".bmp", 10000, "play00001.bmp"),
    ],
)
def test_build_renamed_filename_uses_scene_prefix_and_padding(
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
        - build_renamed_filenameを呼び出す
    Assert:
        - scene名 + ゼロ埋め連番 + 拡張子の形式で返されること
        - 要求枚数が4桁以下なら4桁、それ以上なら要求枚数の桁数でゼロ埋めされること
    """
    # Arrange — パラメタライズド引数からscene名、連番、拡張子、要求枚数を設定
    # Act
    result = FileUtils.build_renamed_filename(
        scene_name=scene_name,
        index=index,
        suffix=suffix,
        requested_num=requested_num,
    )

    # Assert
    assert result == expected


def test_copy_selected_items_rename_avoids_collision_and_counts_per_scene(
    tmp_path: Path,
) -> None:
    """scene別連番で出力しつつ既存ファイルとの衝突を回避できること.

    Arrange:
        - 出力ディレクトリに既存ファイル（play0001.jpg）がある
        - play 2件、event 1件の選択画像がある
    Act:
        - rename=Trueでcopy_selected_itemsを実行する
    Assert:
        - 既存ファイルとの衝突を回避してサフィックス付きで出力されること
        - scene別に連番が振られること
    """
    # Arrange
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "play0001.jpg").write_bytes(b"existing")

    gameplay1 = tmp_path / "play1.jpg"
    gameplay2 = tmp_path / "play2.jpg"
    event1 = tmp_path / "event1.jpg"
    for path in (gameplay1, gameplay2, event1):
        path.write_bytes(b"fake_image_data")

    selected = [
        create_scored_candidate(
            path=str(gameplay1),
            scene_label=SceneLabel.PLAY,
        ),
        create_scored_candidate(
            path=str(event1),
            scene_label=SceneLabel.EVENT,
        ),
        create_scored_candidate(
            path=str(gameplay2),
            scene_label=SceneLabel.PLAY,
        ),
    ]

    # Act
    copied_paths = FileUtils.copy_selected_items(
        selected,
        str(output_dir),
        rename=True,
        requested_num=3,
    )

    # Assert
    assert (output_dir / "play0001_1.jpg").exists()
    assert (output_dir / "event0001.jpg").exists()
    assert (output_dir / "play0002.jpg").exists()
    assert copied_paths[selected[0].path] == str(
        (output_dir / "play0001_1.jpg").resolve()
    )
    assert copied_paths[selected[1].path] == str(
        (output_dir / "event0001.jpg").resolve()
    )
    assert copied_paths[selected[2].path] == str(
        (output_dir / "play0002.jpg").resolve()
    )


def test_copy_selected_items_copies_files(tmp_path: Path) -> None:
    """選択されたアイテムがコピー先にコピーされること.

    Arrange:
        - ソース画像ファイルを作成する
        - コピー先ディレクトリを用意する
        - 候補画像を1件作成する
    Act:
        - copy_selected_itemsを呼び出す
    Assert:
        - 戻り値の件数が1件であること
        - コピー先にファイルが存在すること
    """
    # Arrange
    src_file = tmp_path / "source" / "test.png"
    src_file.parent.mkdir()
    src_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    dest_dir = tmp_path / "output"
    candidate = create_scored_candidate(path=str(src_file))

    # Act
    result = FileUtils.copy_selected_items([candidate], str(dest_dir))

    # Assert
    assert len(result) == 1
    assert (dest_dir / "test.png").exists()


def test_copy_selected_items_returns_path_mapping(tmp_path: Path) -> None:
    """戻り値が path → コピー先パス の対応表であること.

    Arrange:
        - ソース画像ファイルを作成する
        - コピー先ディレクトリを用意する
        - 候補画像を1件作成する
    Act:
        - copy_selected_itemsを呼び出す
    Assert:
        - 戻り値に元パスがキーとして含まれること
        - 値がコピー先パスで元ファイル名で終わること
    """
    # Arrange
    src_file = tmp_path / "source" / "test.png"
    src_file.parent.mkdir()
    src_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    dest_dir = tmp_path / "output"
    candidate = create_scored_candidate(path=str(src_file))

    # Act
    result = FileUtils.copy_selected_items([candidate], str(dest_dir))

    # Assert
    assert candidate.path in result
    assert result[candidate.path].endswith("test.png")


def test_copy_selected_items_renames_by_scene(tmp_path: Path) -> None:
    """rename=True の場合、scene別連番ファイル名が付けられること.

    Arrange:
        - 異なる拡張子の画像ファイルを2件作成する
        - コピー先ディレクトリを用意する
        - 候補画像を2件作成する
    Act:
        - rename=True, requested_num=2 でcopy_selected_itemsを呼び出す
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
    candidates = [
        create_scored_candidate(path=str(tmp_path / "source" / "a.png")),
        create_scored_candidate(path=str(tmp_path / "source" / "b.jpg")),
    ]

    # Act
    result = FileUtils.copy_selected_items(
        candidates, str(dest_dir), rename=True, requested_num=2
    )

    # Assert
    assert len(result) == 2
    files = sorted(dest_dir.iterdir())
    assert files[0].name == "play0001.png"
    assert files[1].name == "play0002.jpg"


def test_copy_selected_items_raises_when_rename_without_requested_num(
    tmp_path: Path,
) -> None:
    """rename=True で requested_num=None の場合、ValueError が送出されること.

    Arrange:
        - ソース画像ファイルを作成する
        - コピー先ディレクトリを用意する
        - 候補画像を1件作成する
    Act:
        - rename=True, requested_num指定なしでcopy_selected_itemsを呼び出す
    Assert:
        - ValueErrorが送出されること
    """
    # Arrange
    src_file = tmp_path / "source" / "test.png"
    src_file.parent.mkdir()
    src_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    dest_dir = tmp_path / "output"
    candidate = create_scored_candidate(path=str(src_file))

    # Act & Assert
    with pytest.raises(ValueError, match="requested_num"):
        FileUtils.copy_selected_items([candidate], str(dest_dir), rename=True)


def test_copy_selected_items_handles_duplicate_filenames(tmp_path: Path) -> None:
    """同名ファイルがある場合、ユニークな名前が生成されること.

    Arrange:
        - 異なるディレクトリに同名の画像ファイルを2件作成する
        - コピー先ディレクトリを用意する
        - 候補画像を2件作成する
    Act:
        - copy_selected_itemsを呼び出す
    Assert:
        - 戻り値が2件であること
        - コピー先に2ファイルが存在すること
        - 一方がtest.png、他方がtest_1.pngであること
    """
    # Arrange
    for i, name in enumerate(["test.png", "test.png"]):
        src_dir = tmp_path / f"source{i}"
        src_dir.mkdir()
        src_file = src_dir / name
        src_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    dest_dir = tmp_path / "output"
    candidates = [
        create_scored_candidate(path=str(tmp_path / "source0" / "test.png")),
        create_scored_candidate(path=str(tmp_path / "source1" / "test.png")),
    ]

    # Act
    result = FileUtils.copy_selected_items(candidates, str(dest_dir))

    # Assert
    assert len(result) == 2
    copied_files = list(dest_dir.iterdir())
    assert len(copied_files) == 2
    names = {f.name for f in copied_files}
    assert "test.png" in names
    assert "test_1.png" in names
