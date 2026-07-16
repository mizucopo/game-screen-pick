"""Context Cue IDの安定導出。"""

import hashlib
import json
from fractions import Fraction

from ..models.context_cue import ContextSourceKind

_ALGORITHM = "context-cue-id-v1"


def build_context_cue_id(
    *,
    video_fingerprint: str,
    source_kind: ContextSourceKind,
    stream_index: int,
    start: Fraction,
    end: Fraction,
    text: str,
) -> str:
    """合意済みcanonical inputから`cue_` IDを返す。"""
    payload = {
        "algorithm": _ALGORITHM,
        "video_fingerprint": video_fingerprint,
        "source_kind": source_kind,
        "stream_index": stream_index,
        "start": [start.numerator, start.denominator],
        "end": [end.numerator, end.denominator],
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return "cue_" + hashlib.sha256(canonical).hexdigest()
