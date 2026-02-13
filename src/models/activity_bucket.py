"""活動量バケット."""

from enum import Enum


class ActivityBucket(Enum):
    """活動量バケット."""

    LOW = "low"
    MID = "mid"
    HIGH = "high"
