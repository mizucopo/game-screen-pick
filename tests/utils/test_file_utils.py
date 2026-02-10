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


# ============================================================================
# 基本機能のテスト
# ============================================================================


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
    """既存ファイルに基づいて一意なファイル名を生成することを検証.

    Given:
        - 出力ディレクトリに既存ファイルが存在（または空）
    When:
        - get_unique_destinationを実行
    Then:
        - 適切なサフィックス付きのファイル名が返される
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


# ============================================================================
# 拡張子維持のテスト
# ============================================================================


@pytest.mark.parametrize(
    "extension",
    [".jpg", ".png", ".gif", ".webp", ".bmp"],
)
def test_preserves_common_image_extensions(
    tmp_path: Path,
    extension: str,
) -> None:
    """一般的な画像形式の拡張子を維持することを検証.

    Given:
        - 重複ファイルが存在する特定の拡張子
    When:
        - get_unique_destinationを実行
    Then:
        - 拡張子が正しく維持される
    """
    # Arrange
    filename = f"image{extension}"
    (tmp_path / filename).touch()

    # Act
    result = FileUtils.get_unique_destination(tmp_path, filename)

    # Assert
    assert result == tmp_path / f"image_1{extension}"
    assert result.suffix == extension


def test_handles_double_extensions(
    tmp_path: Path,
) -> None:
    """複数拡張子を持つファイル名を正しく処理することを検証.

    Given:
        - .tar.gz のような複数拡張子のファイルが存在
    When:
        - get_unique_destinationを実行
    Then:
        - 拡張子が正しく維持される（stemはarchive.tar、suffixは.gzとして処理）
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


# ============================================================================
# エッジケースのテスト
# ============================================================================


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
    """アンダースコア、末尾数字、ハイフンを含むファイル名を正しく処理することを検証.

    Given:
        - 特殊パターンを含むファイル名が存在
    When:
        - get_unique_destinationを実行
    Then:
        - 正しくサフィックスが付与される
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
    """既存の連番にギャップがある場合、次の有効な番号を使用することを検証.

    Given:
        - 連番にギャップがある既存ファイル
    When:
        - get_unique_destinationを実行
    Then:
        - 次の有効な連番が使用される
    """
    # Arrange
    for f in existing_files:
        (tmp_path / f).touch()

    # Act
    result = FileUtils.get_unique_destination(tmp_path, "image.jpg")

    # Assert
    expected = f"image{expected_suffix}.jpg"
    assert result.name == expected


def test_handles_very_long_filename(
    tmp_path: Path,
) -> None:
    """非常に長いファイル名を正しく処理することを検証.

    Given:
        - 長いファイル名の重複が存在
    When:
        - get_unique_destinationを実行
    Then:
        - サフィックスが正しく付与される
        - 拡張子が維持される
    """
    # Arrange
    filename = "a" * 200 + ".jpg"
    (tmp_path / filename).touch()

    # Act
    result = FileUtils.get_unique_destination(tmp_path, filename)

    # Assert
    # サフィックスが付与され、元のファイル名が含まれていることを検証
    assert "_1" in result.name
    assert result.suffix == ".jpg"
    assert "a" * 200 in result.stem


def test_handles_only_stem_no_suffix(
    tmp_path: Path,
) -> None:
    """ステムのみで拡張子がないファイルを正しく処理することを検証.

    Given:
        - 拡張子なしのファイルが存在
    When:
        - get_unique_destinationを実行
    Then:
        - _1サフィックスがステムに直接付与される
    """
    # Arrange
    filename = "myfile"
    (tmp_path / filename).touch()

    # Act
    result = FileUtils.get_unique_destination(tmp_path, filename)

    # Assert
    assert result == tmp_path / "myfile_1"


def test_handles_dotfiles_without_extension(
    tmp_path: Path,
) -> None:
    """ドットで始まる拡張子なしのファイルを正しく処理することを検証.

    Given:
        - .gitignore のようなドットで始まるファイルが存在
    When:
        - get_unique_destinationを実行
    Then:
        - 正しくサフィックスが付与される
    """
    # Arrange
    filename = ".hidden"
    (tmp_path / filename).touch()

    # Act
    result = FileUtils.get_unique_destination(tmp_path, filename)

    # Assert
    assert result == tmp_path / ".hidden_1"


# ============================================================================
# 特殊文字のテスト
# ============================================================================


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("画像ファイル.jpg", "画像ファイル_1.jpg"),
        ("image (copy).jpg", "image (copy)_1.jpg"),
        ("my image file.jpg", "my image file_1.jpg"),
    ],
)
def test_handles_special_characters_in_filename(
    tmp_path: Path,
    filename: str,
    expected: str,
) -> None:
    """特殊文字・日本語・スペースを含むファイル名を正しく処理することを検証.

    Given:
        - 特殊文字、日本語、またはスペースを含むファイル名が存在
    When:
        - get_unique_destinationを実行
    Then:
        - 正しくサフィックスが付与される
    """
    # Arrange
    (tmp_path / filename).touch()

    # Act
    result = FileUtils.get_unique_destination(tmp_path, filename)

    # Assert
    assert result == tmp_path / expected


# ============================================================================
# 境界値のテスト
# ============================================================================


def test_handles_many_duplicate_files(
    tmp_path: Path,
) -> None:
    """多数の重複ファイルが存在する場合、適切な番号を選択することを検証.

    Given:
        - image.jpg から image_99.jpg までが存在
    When:
        - get_unique_destinationを実行
    Then:
        - image_100.jpg が返される
    """
    # Arrange
    filename = "image.jpg"
    (tmp_path / filename).touch()
    for i in range(1, 100):
        (tmp_path / f"image_{i}.jpg").touch()

    # Act
    result = FileUtils.get_unique_destination(tmp_path, filename)

    # Assert
    assert result == tmp_path / "image_100.jpg"
