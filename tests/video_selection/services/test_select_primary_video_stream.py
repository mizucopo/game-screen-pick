"""Primary Video Stream選択のtest。"""

from fractions import Fraction

import pytest

from src.video_selection.models.media_probe import MediaProbe
from src.video_selection.models.media_stream import MediaStream
from src.video_selection.services.select_primary_video_stream import (
    select_primary_video_stream,
)


def _video_stream(
    index: int,
    *,
    is_default: bool = False,
    is_attached_picture: bool = False,
) -> MediaStream:
    return MediaStream(
        index=index,
        kind="video",
        codec_name="ffv1",
        time_base=Fraction(1, 1000),
        start_pts=0,
        duration_ts=1000,
        width=1920,
        height=1080,
        sample_rate=None,
        channels=None,
        language=None,
        is_default=is_default,
        is_forced=False,
        is_attached_picture=is_attached_picture,
    )


def test_default_motion_stream_is_selected_before_lower_index_cover() -> None:
    """静止coverが除外されdefaultの表示映像streamが選択されること。

    Arrange:
        - 最小indexのattached pictureと複数の表示映像streamが用意される
    Act:
        - Primary Video Streamが選択される
    Assert:
        - default dispositionを持つ表示映像streamが返されること
    """
    # Arrange
    probe = MediaProbe(
        format_names=("matroska",),
        streams=(
            _video_stream(0, is_attached_picture=True),
            _video_stream(1),
            _video_stream(2, is_default=True),
        ),
    )

    # Act
    selected = select_primary_video_stream(probe)

    # Assert
    assert selected.index == 2


def test_media_without_motion_video_stream_is_rejected() -> None:
    """表示映像streamがないmediaが拒否されること。

    Arrange:
        - attached pictureだけを持つprobeが用意される
    Act:
        - Primary Video Streamの選択が試行される
    Assert:
        - 表示映像stream不足として失敗すること
    """
    # Arrange
    probe = MediaProbe(
        format_names=("matroska",),
        streams=(_video_stream(0, is_attached_picture=True),),
    )

    # Act / Assert
    with pytest.raises(ValueError, match="Primary Video Stream"):
        select_primary_video_stream(probe)
