"""ffprobe JSON parserのtest。"""

from fractions import Fraction

import pytest

from src.video_selection.media.ffprobe_parser import parse_media_probe


def test_container_duration_is_preserved_as_an_exact_fraction() -> None:
    """container durationの小数値が正確な有理数として保持されること。

    Arrange:
        - 小数durationを持つffprobe documentが用意される
    Act:
        - Media Probeへ変換される
    Assert:
        - durationがfloatへ丸められず保持されること
    """
    # Arrange
    document = {
        "format": {
            "format_name": "matroska,webm",
            "duration": "900.000001",
        },
        "streams": [],
    }

    # Act
    probe = parse_media_probe(document)

    # Assert
    assert probe.duration == Fraction(900000001, 1000000)


def test_non_positive_container_duration_is_rejected() -> None:
    """正でないcontainer durationが不正なprobeとして拒否されること。

    Arrange:
        - durationが0のffprobe documentが用意される
    Act:
        - Media Probeへの変換が試行される
    Assert:
        - 正のdurationが必要であることが通知されること
    """
    # Arrange
    document = {
        "format": {
            "format_name": "matroska",
            "duration": "0.000000",
        },
        "streams": [],
    }

    # Act
    with pytest.raises(ValueError) as exc_info:
        parse_media_probe(document)

    # Assert
    assert "format durationは正" in str(exc_info.value)
