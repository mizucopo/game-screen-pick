"""file_utils.pyの単体テスト.

このテストモジュールは以下のベストプラクティスに従っています：
1. 純粋関数としての動作を検証（ファイルシステム操作なしでテスト可能）
2. AAAパターン（Arrange, Act, Assert）を使用
3. 明確な日本語コメントでテスト意図を説明
4. 拡張子維持、エッジケース、特殊文字、境界値を網羅
"""

from pathlib import Path


from src.utils.file_utils import FileUtils


# ============================================================================
# 基本機能のテスト（3件）
# ============================================================================


def test_returns_original_path_when_no_duplicate(
    tmp_path: Path,
) -> None:
    """重複ファイルが存在しない場合、元のファイル名を返すことを検証.

    Given:
        - 空の出力ディレクトリ
        - 任意のファイル名
    When:
        - get_unique_destinationを実行
    Then:
        - 元のファイル名がそのまま返される
    """
    # Arrange
    filename = "image.jpg"

    # Act
    result = FileUtils.get_unique_destination(tmp_path, filename)

    # Assert
    assert result == tmp_path / filename
    assert result.name == filename


def test_adds_suffix_when_duplicate_exists(
    tmp_path: Path,
) -> None:
    """同名ファイルが存在する場合、連番サフィックスを付与することを検証.

    Given:
        - 出力ディレクトリに同名ファイルが存在
    When:
        - get_unique_destinationを実行
    Then:
        - _1サフィックス付きのファイル名が返される
    """
    # Arrange
    filename = "image.jpg"
    (tmp_path / filename).touch()

    # Act
    result = FileUtils.get_unique_destination(tmp_path, filename)

    # Assert
    assert result == tmp_path / "image_1.jpg"
    assert result.name == "image_1.jpg"


def test_increments_suffix_for_multiple_duplicates(
    tmp_path: Path,
) -> None:
    """複数の重複ファイルが存在する場合、連番をインクリメントすることを検証.

    Given:
        - image.jpg, image_1.jpg, image_2.jpg が存在
    When:
        - get_unique_destinationを実行
    Then:
        - image_3.jpg が返される
    """
    # Arrange
    filename = "image.jpg"
    (tmp_path / filename).touch()
    (tmp_path / "image_1.jpg").touch()
    (tmp_path / "image_2.jpg").touch()

    # Act
    result = FileUtils.get_unique_destination(tmp_path, filename)

    # Assert
    assert result == tmp_path / "image_3.jpg"
    assert result.name == "image_3.jpg"


# ============================================================================
# 拡張子維持のテスト（4件）
# ============================================================================


def test_preserves_common_image_extensions(
    tmp_path: Path,
) -> None:
    """一般的な画像形式の拡張子を維持することを検証.

    Given:
        - 異なる拡張子の重複ファイルが存在
    When:
        - 各拡張子でget_unique_destinationを実行
    Then:
        - 拡張子が正しく維持される
    """
    # Arrange
    extensions = [".jpg", ".png", ".gif", ".webp", ".bmp"]
    for ext in extensions:
        (tmp_path / f"image{ext}").touch()

    # Act & Assert
    for ext in extensions:
        result = FileUtils.get_unique_destination(tmp_path, f"image{ext}")
        assert result == tmp_path / f"image_1{ext}"
        assert result.suffix == ext


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


def test_handles_files_without_extension(
    tmp_path: Path,
) -> None:
    """拡張子なしのファイルを正しく処理することを検証.

    Given:
        - 拡張子なしのファイルが存在
    When:
        - get_unique_destinationを実行
    Then:
        - stemと空のsuffixで正しく処理される
    """
    # Arrange
    filename = "README"
    (tmp_path / filename).touch()

    # Act
    result = FileUtils.get_unique_destination(tmp_path, filename)

    # Assert
    assert result == tmp_path / "README_1"
    assert result.suffix == ""


def test_handles_dot_at_start_of_filename(
    tmp_path: Path,
) -> None:
    """ドットで始まるファイル名を正しく処理することを検証.

    Given:
        - .gitignore のようなドットで始まるファイルが存在
    When:
        - get_unique_destinationを実行
    Then:
        - 正しくサフィックスが付与される
    """
    # Arrange
    filename = ".gitignore"
    (tmp_path / filename).touch()

    # Act
    result = FileUtils.get_unique_destination(tmp_path, filename)

    # Assert
    assert result == tmp_path / ".gitignore_1"
    assert result.suffix == ""


# ============================================================================
# エッジケースのテスト（7件）
# ============================================================================


def test_handles_filename_with_existing_underscore(
    tmp_path: Path,
) -> None:
    """既にアンダースコアを含むファイル名を正しく処理することを検証.

    Given:
        - image_test.jpg のようなファイルが存在
    When:
        - get_unique_destinationを実行
    Then:
        - image_test_1.jpg が返される（stemはimage_test_1）
    """
    # Arrange
    filename = "image_test.jpg"
    (tmp_path / filename).touch()

    # Act
    result = FileUtils.get_unique_destination(tmp_path, filename)

    # Assert
    assert result == tmp_path / "image_test_1.jpg"
    # サフィックスが付与された後のstemは image_test_1 になる
    assert result.stem == "image_test_1"


def test_handles_filename_with_trailing_numbers(
    tmp_path: Path,
) -> None:
    """末尾に数字を含むファイル名を正しく処理することを検証.

    Given:
        - image2.jpg のような末尾数字のファイルが存在
    When:
        - get_unique_destinationを実行
    Then:
        - image2_1.jpg が返される
    """
    # Arrange
    filename = "image2.jpg"
    (tmp_path / filename).touch()

    # Act
    result = FileUtils.get_unique_destination(tmp_path, filename)

    # Assert
    assert result == tmp_path / "image2_1.jpg"


def test_handles_gaps_in_existing_numbered_files(
    tmp_path: Path,
) -> None:
    """既存の連番にギャップがある場合、次の有効な番号を使用することを検証.

    Given:
        - image.jpg と image_5.jpg のみが存在
    When:
        - get_unique_destinationを実行
    Then:
        - image_1.jpg が返される（ギャップを埋めるのではなく、連続的にインクリメント）
    """
    # Arrange
    filename = "image.jpg"
    (tmp_path / filename).touch()
    (tmp_path / "image_5.jpg").touch()

    # Act
    result = FileUtils.get_unique_destination(tmp_path, filename)

    # Assert
    assert result == tmp_path / "image_1.jpg"


def test_handles_non_consecutive_duplicates(
    tmp_path: Path,
) -> None:
    """非連続的な重複ファイルが存在する場合、適切な番号を選択することを検証.

    Given:
        - image.jpg, image_1.jpg, image_3.jpg が存在
    When:
        - get_unique_destinationを実行
    Then:
        - image_2.jpg が返される（次の有効な連番）
    """
    # Arrange
    filename = "image.jpg"
    (tmp_path / filename).touch()
    (tmp_path / "image_1.jpg").touch()
    (tmp_path / "image_3.jpg").touch()

    # Act
    result = FileUtils.get_unique_destination(tmp_path, filename)

    # Assert
    assert result == tmp_path / "image_2.jpg"


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
    """
    # Arrange
    filename = "a" * 200 + ".jpg"
    (tmp_path / filename).touch()

    # Act
    result = FileUtils.get_unique_destination(tmp_path, filename)

    # Assert
    assert result.stem.startswith("a" * 200)
    assert result.stem.endswith("_1")
    assert result.suffix == ".jpg"


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


def test_handles_only_suffix_no_stem(
    tmp_path: Path,
) -> None:
    """ステムがなく拡張子のみのファイルを正しく処理することを検証.

    Given:
        - .gitignore のような拡張子のみのファイルが存在
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
# 特殊文字のテスト（3件）
# ============================================================================


def test_handles_japanese_filename(
    tmp_path: Path,
) -> None:
    """日本語を含むファイル名を正しく処理することを検証.

    Given:
        - 日本語を含むファイル名の重複が存在
    When:
        - get_unique_destinationを実行
    Then:
        - 正しくサフィックスが付与される
    """
    # Arrange
    filename = "画像ファイル.jpg"
    (tmp_path / filename).touch()

    # Act
    result = FileUtils.get_unique_destination(tmp_path, filename)

    # Assert
    assert result == tmp_path / "画像ファイル_1.jpg"


def test_handles_special_characters_in_filename(
    tmp_path: Path,
) -> None:
    """特殊文字を含むファイル名を正しく処理することを検証.

    Given:
        - 特殊文字（スペース、括弧など）を含むファイル名が存在
    When:
        - get_unique_destinationを実行
    Then:
        - 正しくサフィックスが付与される
    """
    # Arrange
    filename = "image (copy).jpg"
    (tmp_path / filename).touch()

    # Act
    result = FileUtils.get_unique_destination(tmp_path, filename)

    # Assert
    assert result == tmp_path / "image (copy)_1.jpg"


def test_handles_spaces_in_filename(
    tmp_path: Path,
) -> None:
    """スペースを含むファイル名を正しく処理することを検証.

    Given:
        - スペースを含むファイル名の重複が存在
    When:
        - get_unique_destinationを実行
    Then:
        - 正しくサフィックスが付与される
    """
    # Arrange
    filename = "my image file.jpg"
    (tmp_path / filename).touch()

    # Act
    result = FileUtils.get_unique_destination(tmp_path, filename)

    # Assert
    assert result == tmp_path / "my image file_1.jpg"


# ============================================================================
# 境界値のテスト（3件）
# ============================================================================


def test_handles_zero_existing_files(
    tmp_path: Path,
) -> None:
    """既存ファイルが0件の場合、元のファイル名を返すことを検証.

    Given:
        - 空の出力ディレクトリ
    When:
        - get_unique_destinationを実行
    Then:
        - 元のファイル名が返される
    """
    # Arrange
    filename = "test.jpg"
    # ディレクトリは空

    # Act
    result = FileUtils.get_unique_destination(tmp_path, filename)

    # Assert
    assert result == tmp_path / filename


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


def test_does_not_modify_existing_files(
    tmp_path: Path,
) -> None:
    """既存のファイルが変更されないことを検証.

    Given:
        - 既存ファイルが存在
    When:
        - get_unique_destinationを実行
    Then:
        - 既存ファイルが変更されていない
    """
    # Arrange
    filename = "image.jpg"
    existing_file = tmp_path / filename
    existing_file.touch()
    original_stat = existing_file.stat()

    # Act
    FileUtils.get_unique_destination(tmp_path, filename)

    # Assert
    # 既存ファイルが変更されていない（統計情報が同じ）
    new_stat = existing_file.stat()
    assert original_stat.st_size == new_stat.st_size
