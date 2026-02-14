"""活動量レベルを表す列挙型."""

from enum import Enum


class ActivityBucket(Enum):
    """画像の活動量レベル（LOW: 低活動, MID: 中活動, HIGH: 高活動）."""

    LOW = "low"
    MID = "mid"
    HIGH = "high"
