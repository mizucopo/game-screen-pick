"""file_utils.pyの単体テスト.

このテストモジュールは以下のベストプラクティスに従っています：
1. 純粋関数としての動作を検証（ファイルシステム操作なしでテスト可能）
2. AAAパターン（Arrange, Act, Assert）を使用
3. 明確な日本語コメントでテスト意図を説明
4. 拡張子維持、エッジケース、特殊文字、境界値を網羅
"""

from pathlib import Path

import pytest

from src.utils.file_utils import FileUtils


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
    assert result.name == expected
