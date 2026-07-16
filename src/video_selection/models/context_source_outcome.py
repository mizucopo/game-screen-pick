"""Context sourceごとの正常な処理結果。"""

from dataclasses import dataclass
from typing import Literal

from .context_cue import ContextSourceKind

ContextSourceStatus = Literal[
    "available",
    "absent",
    "no_context",
    "no_speech",
    "low_reliability",
]


@dataclass(frozen=True)
class ContextSourceOutcome:
    """選択・試行された一つのContext sourceのstatus。"""

    source_kind: ContextSourceKind
    stream_index: int | None
    status: ContextSourceStatus
    reason_code: str
    cue_count: int = 0
    rejected_count: int = 0
    processed_chunk_count: int = 0
