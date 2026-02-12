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
    "existing_files,filename,expected",
    [
        # 基本的なケース
        ([], "image.jpg", "image.jpg"),
        (["image.jpg"], "image.jpg", "image_1.jpg"),
        (["image.jpg", "image_1.jpg", "image_2.jpg"], "image.jpg", "image_3.jpg"),
        # ギャップがある場合
        (["image.jpg", "image_1.jpg", "image_3.jpg"], "image.jpg", "image_2.jpg"),
        (["image.jpg", "image_5.jpg"], "image.jpg", "image_1.jpg"),
        # 特殊パターンを含むファイル名
        (["image_test.jpg"], "image_test.jpg", "image_test_1.jpg"),
        (["image2.jpg"], "image2.jpg", "image2_1.jpg"),
        (["image-test.jpg"], "image-test.jpg", "image-test_1.jpg"),
        # 複数拡張子
        (["archive.tar.gz"], "archive.tar.gz", "archive.tar_1.gz"),
        # 特殊文字・日本語・スペース
        (["画像ファイル.jpg"], "画像ファイル.jpg", "画像ファイル_1.jpg"),
        (["image (copy).jpg"], "image (copy).jpg", "image (copy)_1.jpg"),
        (["my image file.jpg"], "my image file.jpg", "my image file_1.jpg"),
        # 拡張子なし・ドットファイル
        (["myfile"], "myfile", "myfile_1"),
        ([".hidden"], ".hidden", ".hidden_1"),
        # 非常に長いファイル名
        (["a" * 200 + ".jpg"], "a" * 200 + ".jpg", "a" * 200 + "_1.jpg"),
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
    assert result.name == expected
