"""確定済みProcessing Stage。"""

from collections.abc import Mapping
from dataclasses import dataclass, field

from .processing_stage import ProcessingStage
from .stage_fingerprint import StageFingerprint


@dataclass(frozen=True)
class CompletedStage:
    """artifactとmanifestがatomicに確定したStage。"""

    stage: ProcessingStage
    fingerprint: StageFingerprint
    upstream_fingerprints: tuple[StageFingerprint, ...] = ()
    semantic_input: Mapping[str, object] = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )
