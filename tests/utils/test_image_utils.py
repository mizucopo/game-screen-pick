"""image_utils.pyの単体テスト."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.utils.image_utils import ImageUtils


@pytest.mark.parametrize(
    "size,max_dim,expected_max_size",
    [
        # 大きな画像は縮小される
        ((2000, 1000), 720, 720),
        # 小さな画像はそのまま
        ((300, 200), 720, 300),
        # グレースケールもRGBに変換
        ((100, 100), 720, 100),
    ],
)
def test_load_as_rgb_resized(
    tmp_path: Path,
    size: tuple[int, int],
    max_dim: int,
    expected_max_size: int,
) -> None:
    """画像が正しくRGB形式でリサイズされて読み込まれること.

    Arrange:
        - 指定されたサイズの画像がある
    Act:
        - load_as_rgb_resizedで読み込む
    Assert:
        - RGB形式の画像が返されること
        - 長辺がmax_dim以下であること
    """
    # Arrange
    channels = 1 if len(size) == 2 and size == (100, 100) else 3
    shape = (*size, channels) if channels == 3 else size
    img_array = np.random.randint(0, 255, shape, dtype=np.uint8)

    suffix = "_gray.jpg" if channels == 1 else ".jpg"
    image_path = tmp_path / f"test{suffix}"
    cv2.imwrite(str(image_path), img_array)

    # Act
    result = ImageUtils.load_as_rgb_resized(str(image_path), max_dim=max_dim)

    # Assert
    assert result is not None
    assert result.mode == "RGB"

    w, h = result.size
    max_size = max(w, h)
    assert max_size == expected_max_size

    # アスペクト比が保持されている
    original_aspect = size[1] / size[0] if size[0] > 0 else 1.0
    result_aspect = w / h if h > 0 else 1.0
    assert abs(original_aspect - result_aspect) < 0.01


def test_load_as_rgb_resized_returns_none_for_invalid_path() -> None:
    """無効なパスでNoneが返されること.

    Arrange:
        - 存在しないファイルパスが指定されている
    Act:
        - load_as_rgb_resizedが実行される
    Assert:
        - Noneが返されること
    """
    # Arrange
    invalid_path = "/nonexistent/path/image.jpg"

    # Act
    result = ImageUtils.load_as_rgb_resized(invalid_path, max_dim=720)

    # Assert
    assert result is None
