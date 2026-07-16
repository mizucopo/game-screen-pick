"""Context Cueに採用されなかったSTT診断。"""

from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from .context_cue_provenance import ContextCueProvenance


@dataclass(frozen=True)
class RejectedSpeechDiagnostic:
    """processing cache内だけに保持する低reliability文字列。"""

    stream_index: int
    start: Fraction
    end: Fraction
    text: str
    average_log_probability: float
    no_speech_probability: float | None
    word_probabilities: tuple[float | None, ...]
    reason_code: str = "asr_below_policy_threshold"
    reliability: Literal["low"] = "low"
    provenance: ContextCueProvenance | None = None
