"""Selection profile definitions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SelectionProfile:
    """選定プロファイル定義."""

    name: str
    quality_weights: dict[str, float]
    activity_weights: dict[str, float]
    activity_mix_ratio: tuple[float, float, float]
    selection_scene_weight: float
    selection_quality_weight: float
