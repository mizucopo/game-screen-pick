"""Context Cueのsource provenance。"""

from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

ContextLanguageSource = Literal["stream_metadata", "speech_recognition"]


@dataclass(frozen=True)
class ContextCueProvenance:
    """source timing、stream、STT実行identityを保持する。"""

    codec_name: str
    source_pts: int
    source_time_base: Fraction
    stream_language: str | None
    is_default: bool
    is_forced: bool
    language_source: ContextLanguageSource
    chunk_sample_start: int | None = None
    chunk_sample_end: int | None = None
    speech_runtime_identity: str | None = None
    resolved_model_identity: str | None = None
    device: str | None = None
    compute_type: str | None = None
