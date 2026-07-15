"""walking skeletonのContext Cue。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ContextCue:
    """fake SpeechRuntimeが返す最小Context Cue。"""

    identifier: str
