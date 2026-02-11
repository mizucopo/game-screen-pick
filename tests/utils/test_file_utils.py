"""file_utils.pyの単体テスト.

このテストモジュールは以下のベストプラクティスに従っています：
1. 純粋関数としての動作を検証（ファイルシステム操作なしでテスト可能）
2. AAAパターン（Arrange, Act, Assert）を使用
3. 明確な日本語コメントでテスト意図を説明
4. 拡張子維持、エッジケース、特殊文字、境界値を網羅
"""

import pytest
from pathlib import Path


from src.utils.file_utils import FileUtils


@pytest.mark.parametrize(
    "existing_files,expected_suffix",
    [
        (["image.jpg"], "_1"),
        (["image.jpg", "image_1.jpg", "image_2.jpg"], "_3"),
        ([], ""),  # ファイルが存在しない場合はサフィックスなし
    ],
)
def test_get_unique_destination_returns_correct_filename(
    tmp_path: Path,
    existing_files: list[str],
    expected_suffix: str,
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
    result = FileUtils.get_unique_destination(tmp_path, "image.jpg")

    # Assert
    expected_name = f"image{expected_suffix}.jpg"
    assert result == tmp_path / expected_name
    assert result.name == expected_name


def test_handles_double_extensions(
    tmp_path: Path,
) -> None:
    """複数拡張子を持つファイル名が正しく処理されること.

    Given:
        - .tar.gz のような複数拡張子のファイルが存在する
    When:
        - get_unique_destinationが実行される
    Then:
        - 拡張子が正しく維持されること（stemはarchive.tar、suffixは.gzとして処理）
    """
    # Arrange
    filename = "archive.tar.gz"
    (tmp_path / filename).touch()

    # Act
    result = FileUtils.get_unique_destination(tmp_path, filename)

    # Assert
    # Path.suffix は最後の拡張子のみを返すため、archive.tarがstem、.gzがsuffixになる
    assert result == tmp_path / "archive.tar_1.gz"
    assert result.suffix == ".gz"


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("image_test.jpg", "image_test_1.jpg"),
        ("image2.jpg", "image2_1.jpg"),
        ("image-test.jpg", "image-test_1.jpg"),
    ],
)
def test_handles_filenames_with_patterns(
    tmp_path: Path,
    filename: str,
    expected: str,
) -> None:
    """アンダースコア、末尾数字、ハイフンを含むファイル名が正しく処理されること.

    Given:
        - 特殊パターンを含むファイル名が存在する
    When:
        - get_unique_destinationが実行される
    Then:
        - 正しくサフィックスが付与されること
    """
    # Arrange
    (tmp_path / filename).touch()

    # Act
    result = FileUtils.get_unique_destination(tmp_path, filename)

    # Assert
    assert result.name == expected


@pytest.mark.parametrize(
    "existing_files,expected_suffix",
    [
        (["image.jpg", "image_1.jpg", "image_3.jpg"], "_2"),  # ギャップあり
        (["image.jpg", "image_5.jpg"], "_1"),  # 大きなギャップ
    ],
)
def test_handles_gaps_in_numbered_files(
    tmp_path: Path,
    existing_files: list[str],
    expected_suffix: str,
) -> None:
    """既存の連番にギャップがある場合、次の有効な番号が使用されること.

    Given:
        - 連番にギャップがある既存ファイルがある
    When:
        - get_unique_destinationが実行される
    Then:
        - 次の有効な連番が使用されること
    """
    # Arrange
    for f in existing_files:
        (tmp_path / f).touch()

    # Act
    result = FileUtils.get_unique_destination(tmp_path, "image.jpg")

    # Assert
    expected = f"image{expected_suffix}.jpg"
    assert result.name == expected


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("画像ファイル.jpg", "画像ファイル_1.jpg"),
        ("image (copy).jpg", "image (copy)_1.jpg"),
        ("my image file.jpg", "my image file_1.jpg"),
        ("myfile", "myfile_1"),  # 拡張子なし
        (".hidden", ".hidden_1"),  # ドットファイル（拡張子なし）
        ("a" * 200 + ".jpg", "a" * 200 + "_1.jpg"),  # 非常に長いファイル名
    ],
)
def test_handles_special_characters_and_edge_cases(
    tmp_path: Path,
    filename: str,
    expected: str,
) -> None:
    """特殊文字・日本語・スペース・拡張子なし・長いファイル名が正しく処理されること.

    Given:
        - 特殊文字、日本語、スペース、拡張子なし、または非常に長いファイル名が存在する
    When:
        - get_unique_destinationが実行される
    Then:
        - 正しくサフィックスが付与されること
    """
    # Arrange
    (tmp_path / filename).touch()

    # Act
    result = FileUtils.get_unique_destination(tmp_path, filename)

    # Assert
    assert result == tmp_path / expected
