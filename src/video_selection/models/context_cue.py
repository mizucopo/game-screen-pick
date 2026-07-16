"""Video Time区間に対応付けられたContext Cue。"""

from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from .context_cue_diagnostics import ContextCueDiagnostics
from .context_cue_provenance import ContextCueProvenance

ContextSourceKind = Literal["embedded_subtitle", "speech_to_text"]
ContextTimestampBasis = Literal[
    "source_pts",
    "container_text_ms",
    "asr_sample_grid_estimate",
]
ContextCueReliability = Literal["usable"]


@dataclass(frozen=True)
class ContextCue:
    """cache内だけに本文を保持する時間付き文脈。"""

    identifier: str
    video_fingerprint: str = ""
    source_kind: ContextSourceKind = "embedded_subtitle"
    stream_index: int = 0
    start: Fraction = Fraction(0)
    end: Fraction = Fraction(1)
    timestamp_basis: ContextTimestampBasis = "source_pts"
    text: str = ""
    language: str | None = None
    reliability: ContextCueReliability = "usable"
    diagnostics: ContextCueDiagnostics | None = None
    provenance: ContextCueProvenance | None = None
