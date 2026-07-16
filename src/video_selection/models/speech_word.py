"""Speech Runtimeが返す一つのword。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SpeechWord:
    """chunk内の整数PCM sample位置を持つword。"""

    text: str
    start_sample: int
    end_sample: int
    probability: float | None = None
