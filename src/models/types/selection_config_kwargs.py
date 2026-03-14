"""SelectionConfig関連の型定義."""

from typing import TypedDict

from ..scene_mix import SceneMix


class SelectionConfigKwargs(TypedDict, total=False):
    """SelectionConfig.from_cli_args の引数型."""

    batch_size: int
    profile: str
    similarity_threshold: float
    scene_mix: SceneMix
    threshold_relaxation_steps: list[float]
    max_threshold: float
