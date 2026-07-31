"""Speech Runtimeが返す一つのword。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SpeechWord:
    """chunk内の整数PCM sample境界を持つword token。

    backendの時刻量子化でstartとendが同じtokenは保持するが、Context Cue全体には
    正の時間幅を要求する。
    """

    text: str
    start_sample: int
    end_sample: int
    probability: float | None = None
