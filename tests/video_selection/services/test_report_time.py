"""Canonical report exact time projectionのtest。"""

from fractions import Fraction

from src.video_selection.services.report_time import (
    display_report_time,
    exact_seconds_string,
    rational_report_value,
)


def test_display_time_uses_half_up_without_wrapping_at_24_hours() -> None:
    """24時間超のexact秒がhalf-upで折り返さず表示されること。

    Arrange:
        - 25時間と59.9995秒を表す既約分数が用意される
    Act:
        - Report Video Time表示へ投影される
    Assert:
        - millisecondがhalf-upされ25時間を維持した表示になること
    """
    # Arrange
    value = Fraction(25 * 3600 + 59) + Fraction(9995, 10000)

    # Act
    result = display_report_time(value)

    # Assert
    assert result == "25:01:00.000"


def test_exact_seconds_keeps_integer_zeroes_and_nonterminating_fraction() -> None:
    """exact secondsが整数zeroを失わず非有限小数を分数で保持すること。

    Arrange:
        - 10秒、1.25秒、1/3秒が用意される
    Act:
        - losslessなJSON文字列表現へ投影される
    Assert:
        - 整数、有限小数、既約分数として正確に返されること
    """
    # Arrange
    values = (Fraction(10), Fraction(5, 4), Fraction(1, 3))

    # Act
    results = tuple(exact_seconds_string(value) for value in values)

    # Assert
    assert results == ("10", "1.25", "1/3")


def test_rational_value_is_reduced_by_fraction_boundary() -> None:
    """report rational objectへ既約な分子と分母が渡されること。

    Arrange:
        - 約分可能な入力から作られたFractionが用意される
    Act:
        - report rational objectへ投影される
    Assert:
        - 既約な分子と正の分母が返されること
    """
    # Arrange
    value = Fraction(6, 8)

    # Act
    result = rational_report_value(value)

    # Assert
    assert result == {"numerator": 3, "denominator": 4}
