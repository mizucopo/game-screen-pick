"""SelectionConfig関連の型定義."""

from typing import TypedDict


class SelectionConfigKwargs(TypedDict, total=False):
    """SelectionConfig.from_cli_args の引数型."""

    batch_size: int
    threshold_relaxation_steps: list[float]
    max_threshold: float
    activity_mix_enabled: bool
    activity_mix_ratio: tuple[float, float, float]
    activity_bucket_mode: str
