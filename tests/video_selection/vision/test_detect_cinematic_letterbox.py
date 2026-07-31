"""映画的な上下黒帯の決定的検知。"""

import io

import pytest
from PIL import Image, ImageDraw

from src.video_selection.vision.detect_cinematic_letterbox import (
    has_cinematic_letterbox,
)


@pytest.mark.parametrize(
    ("top_bar", "bottom_bar", "expected"),
    (
        (18, 18, True),
        (18, 0, False),
        (0, 18, False),
        (5, 5, False),
    ),
)
def test_only_thick_dark_bands_on_both_edges_are_detected(
    top_bar: int,
    bottom_bar: int,
    expected: bool,
) -> None:
    """上下両端に十分な黒帯がある画像だけ検出されること。

    Arrange:
        - 上下の黒帯幅が異なるJPEG画像が用意される
    Act:
        - 映画的な上下黒帯が検知される
    Assert:
        - 両端に十分な黒帯がある場合だけtrueが返されること
    """
    # Arrange
    image_bytes = _image_bytes(top_bar=top_bar, bottom_bar=bottom_bar)

    # Act
    actual = has_cinematic_letterbox(image_bytes)

    # Assert
    assert actual is expected


def test_uniformly_dark_image_is_not_detected_as_letterboxed() -> None:
    """内容領域のない暗転画像が上下黒帯として扱われないこと。

    Arrange:
        - 画像全体が黒いJPEG画像が用意される
    Act:
        - 映画的な上下黒帯が検知される
    Assert:
        - falseが返されること
    """
    # Arrange
    output = io.BytesIO()
    Image.new("RGB", (320, 180), color=(0, 0, 0)).save(
        output,
        format="JPEG",
        quality=95,
    )

    # Act
    actual = has_cinematic_letterbox(output.getvalue())

    # Assert
    assert actual is False


def test_invalid_image_is_not_detected_as_letterboxed() -> None:
    """画像でないbytesが上下黒帯として扱われないこと。

    Arrange:
        - 画像としてdecodeできないbytesが用意される
    Act:
        - 映画的な上下黒帯が検知される
    Assert:
        - falseが返されること
    """
    # Arrange
    image_bytes = b"not-an-image"

    # Act
    actual = has_cinematic_letterbox(image_bytes)

    # Assert
    assert actual is False


def _image_bytes(*, top_bar: int, bottom_bar: int) -> bytes:
    image = Image.new("RGB", (320, 180), color=(72, 104, 136))
    draw = ImageDraw.Draw(image)
    if top_bar:
        draw.rectangle((0, 0, image.width, top_bar - 1), fill=(0, 0, 0))
    if bottom_bar:
        draw.rectangle(
            (0, image.height - bottom_bar, image.width, image.height),
            fill=(0, 0, 0),
        )
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=95)
    return output.getvalue()
