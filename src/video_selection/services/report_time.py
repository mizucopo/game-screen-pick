"""Canonical reportのexact time projection。"""

import math
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from fractions import Fraction


def display_report_time(value: Fraction) -> str:
    """非負のexact秒をhalf-upでunbounded hour表示する。"""
    if value < 0:
        msg = "Report Video Timeは0以上である必要があります"
        raise ValueError(msg)
    milliseconds = math.floor(value * 1000 + Fraction(1, 2))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def rational_report_value(value: Fraction) -> dict[str, int]:
    """既約FractionをJSON objectへ投影する。"""
    return {"numerator": value.numerator, "denominator": value.denominator}


def exact_seconds_string(value: Fraction) -> str:
    """有限小数は小数、それ以外は既約分数としてlosslessに表示する。"""
    denominator = value.denominator
    reduced = denominator
    for factor in (2, 5):
        while reduced % factor == 0:
            reduced //= factor
    if reduced != 1:
        return f"{value.numerator}/{value.denominator}"
    with localcontext() as context:
        context.prec = max(len(str(abs(value.numerator))), len(str(denominator))) + 8
        result = Decimal(value.numerator) / Decimal(denominator)
    rendered = format(result, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def utc_report_datetime(value: datetime) -> str:
    """timezone-aware datetimeをUTCのZ表記へ正規化する。"""
    if value.tzinfo is None:
        msg = "Report datetimeにはtimezoneが必要です"
        raise ValueError(msg)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
