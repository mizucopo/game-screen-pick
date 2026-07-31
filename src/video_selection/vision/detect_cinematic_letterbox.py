"""画像の画素から映画的な上下黒帯を決定的に検知する。"""

import io
from typing import cast

from PIL import Image

CINEMATIC_LETTERBOX_DETECTION_VERSION = "cinematic-letterbox-detection-v1"

_DARK_CHANNEL_MAX = 24
_DARK_SAMPLE_PERCENT = 95
_MINIMUM_BAR_HEIGHT_PERCENT = 4
_MAXIMUM_PROBE_HEIGHT_PERCENT = 25
_SAMPLE_COLUMN_COUNT = 128


def has_cinematic_letterbox(image_bytes: bytes) -> bool:
    """上下両端の太い暗色帯と、その間の可視内容を検知する。"""
    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            image = source.convert("RGB")
    except (OSError, ValueError):
        return False
    width, height = image.size
    if width < 16 or height < 16:
        return False
    columns = _sample_positions(width)
    minimum_bar_height = max(
        2,
        (height * _MINIMUM_BAR_HEIGHT_PERCENT + 99) // 100,
    )
    maximum_probe_height = max(
        minimum_bar_height,
        height * _MAXIMUM_PROBE_HEIGHT_PERCENT // 100,
    )
    top_depth = _dark_edge_depth(
        image,
        columns,
        range(maximum_probe_height),
    )
    bottom_depth = _dark_edge_depth(
        image,
        columns,
        range(height - 1, height - maximum_probe_height - 1, -1),
    )
    if top_depth < minimum_bar_height or bottom_depth < minimum_bar_height:
        return False
    return any(
        not _row_is_dark(image, columns, row)
        for row in (height * 2 // 5, height // 2, height * 3 // 5)
    )


def _sample_positions(width: int) -> tuple[int, ...]:
    count = min(width, _SAMPLE_COLUMN_COUNT)
    if count == 1:
        return (0,)
    return tuple(index * (width - 1) // (count - 1) for index in range(count))


def _dark_edge_depth(
    image: Image.Image,
    columns: tuple[int, ...],
    rows: range,
) -> int:
    depth = 0
    for row in rows:
        if not _row_is_dark(image, columns, row):
            break
        depth += 1
    return depth


def _row_is_dark(
    image: Image.Image,
    columns: tuple[int, ...],
    row: int,
) -> bool:
    dark_count = sum(
        max(cast(tuple[int, int, int], image.getpixel((column, row))))
        <= _DARK_CHANNEL_MAX
        for column in columns
    )
    return dark_count * 100 >= len(columns) * _DARK_SAMPLE_PERCENT
