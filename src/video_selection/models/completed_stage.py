"""確定済みProcessing Stage。"""

from dataclasses import dataclass

from .processing_stage import ProcessingStage
from .stage_fingerprint import StageFingerprint


@dataclass(frozen=True)
class CompletedStage:
    """artifactとmanifestがatomicに確定したStage。"""

    stage: ProcessingStage
    fingerprint: StageFingerprint
