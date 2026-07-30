"""ReleaseInterval modelのtest。"""

from fractions import Fraction

import pytest

from src.video_selection.acceptance.release_interval import ReleaseInterval


def test_positive_private_interval_exposes_expected_duration() -> None:
    """安全な相対pathと正の区間から期待durationが返されること。

    Arrange:
        - 10秒から70秒のprivate source intervalが用意される
    Act:
        - ReleaseIntervalが構築される
    Assert:
        - 60秒のexpected durationが返されること
    """
    # Arrange
    source_path = "chapter/video.mkv"

    # Act
    interval = ReleaseInterval(source_path, Fraction(10), Fraction(70), "event")

    # Assert
    assert interval.expected_duration == Fraction(60)


def test_parent_segment_is_rejected() -> None:
    """input root外へ出るparent segmentが拒否されること。

    Arrange:
        - parent segmentを持つsource pathが用意される
    Act:
        - ReleaseIntervalの構築が試行される
    Assert:
        - path contract違反としてValueErrorになること
    """
    # Arrange
    source_path = "../video.mkv"

    # Act
    with pytest.raises(ValueError) as error:
        ReleaseInterval(source_path, Fraction(0), Fraction(1), "event")

    # Assert
    assert "Release interval" in str(error.value)
