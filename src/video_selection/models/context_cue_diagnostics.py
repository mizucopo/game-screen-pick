"""採用Context Cueの未校正backend diagnostics。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ContextCueDiagnostics:
    """STT backendが返した比較用途ではないraw値。"""

    average_log_probability: float
    no_speech_probability: float | None
    word_probabilities: tuple[float | None, ...]
