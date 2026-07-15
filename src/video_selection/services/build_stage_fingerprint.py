"""Stage Fingerprintを構築する。"""

import hashlib
import json
from collections.abc import Mapping, Sequence

from ..models.processing_stage import ProcessingStage
from ..models.stage_fingerprint import StageFingerprint
from .stage_version import stage_version


def build_stage_fingerprint(
    stage: ProcessingStage,
    upstream_fingerprints: Sequence[StageFingerprint],
    semantic_input: Mapping[str, object],
) -> StageFingerprint:
    """Stage固有入力と上流fingerprintから安定した値を返す。"""
    normalized = json.dumps(
        {
            "stage": stage.value,
            "stage_version": stage_version(stage),
            "upstream": [item.value for item in upstream_fingerprints],
            "input": semantic_input,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return StageFingerprint(hashlib.sha256(normalized).hexdigest())
