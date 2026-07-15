"""Processing Stageのfingerprint。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class StageFingerprint:
    """一つのProcessing Stage入力を識別する値。"""

    value: str
