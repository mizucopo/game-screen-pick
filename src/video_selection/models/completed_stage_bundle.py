"""検証済みCompleted Stageの成果物bundle。"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CompletedStageBundle:
    """JSON artifactと検証済みartifact rootをまとめる。"""

    artifact: dict[str, object]
    root: Path
