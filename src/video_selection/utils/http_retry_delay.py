"""HTTP Retry-Afterを安全な待機秒へ変換する。"""

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from math import isfinite


def http_retry_delay(status_code: int, retry_after: str | None) -> float:
    """429のdelta-secondsまたはHTTP-dateを0〜30秒へ制限して返す。"""
    if status_code != 429:
        return 1.0
    try:
        seconds = float(retry_after) if retry_after is not None else 1.0
    except ValueError:
        try:
            retry_at = (
                parsedate_to_datetime(retry_after) if retry_after is not None else None
            )
        except (TypeError, ValueError, OverflowError):
            return 1.0
        if retry_at is None:
            return 1.0
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        seconds = (retry_at - datetime.now(timezone.utc)).total_seconds()
    if not isfinite(seconds):
        return 1.0
    return min(max(seconds, 0.0), 30.0)
