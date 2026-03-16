"""Selection profile definitions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SelectionProfile:
    """選定プロファイル定義."""

    name: str
    quality_weights: dict[str, float]
