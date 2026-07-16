"""Speech Runtimeが返す一つのsource segment。"""

from dataclasses import dataclass

from .speech_word import SpeechWord


@dataclass(frozen=True)
class SpeechSegment:
    """word列と未校正backend diagnosticを保持する。"""

    words: tuple[SpeechWord, ...]
    average_log_probability: float
    no_speech_probability: float | None = None
