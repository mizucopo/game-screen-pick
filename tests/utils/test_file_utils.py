"""file_utils.pyの単体テスト.

このテストモジュールは以下のベストプラクティスに従っています：
1. 純粋関数としての動作を検証（ファイルシステム操作なしでテスト可能）
2. AAAパターン（Arrange, Act, Assert）を使用
3. 明確な日本語コメントでテスト意図を説明
4. 拡張子維持、エッジケース、特殊文字、境界値を網羅
"""

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

    Given:
        - 出力ディレクトリに既存ファイルが存在する（または空である）
    When:
        - get_unique_destinationが実行される
    Then:
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
        ("gameplay", 1, ".jpg", 3, "gameplay0001.jpg"),
        ("event", 12, ".png", 9999, "event0012.png"),
        ("other", 1, ".bmp", 10000, "other00001.bmp"),
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

    Given:
        - scene名、連番index、拡張子、要求枚数がある
    When:
        - build_renamed_filenameを呼び出す
    Then:
        - scene名 + ゼロ埋め連番 + 拡張子の形式で返されること
        - 要求枚数が4桁以下なら4桁、それ以上なら要求枚数の桁数でゼロ埋めされること
    """
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

    Given:
        - 出力ディレクトリに既存ファイル（gameplay0001.jpg）がある
        - gameplay 2件、event 1件の選択画像がある
    When:
        - rename=Trueでcopy_selected_itemsを実行する
    Then:
        - 既存ファイルとの衝突を回避してサフィックス付きで出力されること
        - scene別に連番が振られること
    """
    # Arrange
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "gameplay0001.jpg").write_bytes(b"existing")

    gameplay1 = tmp_path / "gameplay1.jpg"
    gameplay2 = tmp_path / "gameplay2.jpg"
    event1 = tmp_path / "event1.jpg"
    for path in (gameplay1, gameplay2, event1):
        path.write_bytes(b"fake_image_data")

    selected = [
        create_scored_candidate(
            path=str(gameplay1),
            scene_label=SceneLabel.GAMEPLAY,
        ),
        create_scored_candidate(
            path=str(event1),
            scene_label=SceneLabel.EVENT,
        ),
        create_scored_candidate(
            path=str(gameplay2),
            scene_label=SceneLabel.GAMEPLAY,
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
    assert (output_dir / "gameplay0001_1.jpg").exists()
    assert (output_dir / "event0001.jpg").exists()
    assert (output_dir / "gameplay0002.jpg").exists()
    assert copied_paths[id(selected[0])] == str(
        (output_dir / "gameplay0001_1.jpg").resolve()
    )
    assert copied_paths[id(selected[1])] == str((output_dir / "event0001.jpg").resolve())
    assert copied_paths[id(selected[2])] == str(
        (output_dir / "gameplay0002.jpg").resolve()
    )
