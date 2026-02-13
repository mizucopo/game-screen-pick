"""image_utils.pyの単体テスト.

このテストモジュールは以下のベストプラクティスに従っています：
1. 純粋関数としての動作を検証
2. AAAパターン（Arrange, Act, Assert）を使用
3. 明確な日本語コメントでテスト意図を説明
4. エッジケース、境界値を網羅
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.utils.image_utils import ImageUtils


def test_load_as_rgb_resized_shrinks_large_images(tmp_path: Path) -> None:
    """大きな画像がmax_dim以下に縮小されること.

    Given:
        - 2000x1000の画像がある
        - max_dim=720で読み込む
    When:
        - load_as_rgb_resizedで読み込む
    Then:
        - 長辺が720以下であること
        - アスペクト比が保持されていること
    """
    # Arrange: 2000x1000の画像を作成
    img_array = np.random.randint(0, 255, (1000, 2000, 3), dtype=np.uint8)
    large_image_path = tmp_path / "large_image.jpg"
    cv2.imwrite(str(large_image_path), img_array)

    # Act: max_dim=720で読み込み
    result = ImageUtils.load_as_rgb_resized(str(large_image_path), max_dim=720)

    # Assert
    assert result is not None
    w, h = result.size
    max_size = max(w, h)
    assert max_size <= 720
    # アスペクト比が保持されている（元は2:1）
    assert abs(w / h - 2.0) < 0.01


def test_load_as_rgb_resized_preserves_small_images(tmp_path: Path) -> None:
    """小さな画像がそのまま返されること.

    Given:
        - 300x200の画像がある
        - max_dim=720で読み込む
    When:
        - load_as_rgb_resizedで読み込む
    Then:
        - 画像サイズが変更されていないこと
    """
    # Arrange: 300x200の画像を作成
    img_array = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
    small_image_path = tmp_path / "small_image.jpg"
    cv2.imwrite(str(small_image_path), img_array)

    # Act: max_dim=720で読み込み（画像が小さいので縮小されないはず）
    result = ImageUtils.load_as_rgb_resized(str(small_image_path), max_dim=720)

    # Assert
    assert result is not None
    w, h = result.size
    assert w == 300
    assert h == 200


def test_load_as_rgb_resized_converts_to_rgb(tmp_path: Path) -> None:
    """グレースケール画像がRGBに変換されること.

    Given:
        - グレースケール画像がある
    When:
        - load_as_rgb_resizedで読み込む
    Then:
        - RGB形式の画像が返されること
    """
    # Arrange: グレースケール画像を作成
    img_array = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    gray_image_path = tmp_path / "gray_image.jpg"
    cv2.imwrite(str(gray_image_path), img_array)

    # Act
    result = ImageUtils.load_as_rgb_resized(str(gray_image_path), max_dim=720)

    # Assert
    assert result is not None
    assert result.mode == "RGB"
    assert result.size == (100, 100)


def test_load_as_rgb_resized_returns_none_on_invalid_path() -> None:
    """無効なパスでNoneが返されること.

    Given:
        - 存在しないファイルパスがある
    When:
        - load_as_rgb_resizedで読み込む
    Then:
        - Noneが返されること
    """
    # Arrange
    invalid_path = "/nonexistent/path/image.jpg"

    # Act
    result = ImageUtils.load_as_rgb_resized(invalid_path, max_dim=720)

    # Assert
    assert result is None


def test_load_as_rgb_resized_default_max_dim(tmp_path: Path) -> None:
    """デフォルトのmax_dim(720)が使用されること.

    Given:
        - 1000x1000の画像がある
    When:
        - max_dimを指定せずにload_as_rgb_resizedで読み込む
    Then:
        - 長辺が720以下であること
    """
    # Arrange: 1000x1000の画像を作成
    img_array = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
    image_path = tmp_path / "square_image.jpg"
    cv2.imwrite(str(image_path), img_array)

    # Act: max_dimを指定しない（デフォルト720）
    result = ImageUtils.load_as_rgb_resized(str(image_path))

    # Assert
    assert result is not None
    w, h = result.size
    assert max(w, h) <= 720


def test_pil_to_cv2_uses_asarray() -> None:
    """pil_to_cv2でnp.asarrayが使用されていること（実装詳細）.

    Given:
        - RGB形式のPIL画像がある
    When:
        - pil_to_cv2で変換する
    Then:
        - BGR形式のNumPy配列が返されること
    """
    # Arrange
    pil_img = Image.new("RGB", (100, 50), color=(255, 0, 0))

    # Act
    result = ImageUtils.pil_to_cv2(pil_img)

    # Assert
    assert isinstance(result, np.ndarray)
    assert result.shape == (50, 100, 3)  # OpenCVは (height, width, channels)
    # PILのRGB(255,0,0)はOpenCVのBGR(0,0,255)になる
    # result[0,0]はBGRなので [0, 0, 255] になるはず
    assert np.array_equal(result[0, 0], [0, 0, 255])
