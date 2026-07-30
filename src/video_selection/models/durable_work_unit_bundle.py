"""検証済みDurable Work Unit checkpoint。"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DurableWorkUnitBundle:
    """checkpointのJSON artifactとartifact rootを保持する。"""

    artifact: dict[str, object]
    root: Path
