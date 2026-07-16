"""backend非依存のSpeech Recognition Result。"""

from dataclasses import dataclass

from .speech_segment import SpeechSegment


@dataclass(frozen=True)
class SpeechRecognitionResult:
    """VAD結果、言語、segment列を返すinfra-level結果。"""

    vad_speech_detected: bool
    segments: tuple[SpeechSegment, ...]
    detected_language: str | None = None
