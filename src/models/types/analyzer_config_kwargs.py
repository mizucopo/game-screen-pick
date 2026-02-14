"""AnalyzerConfig関連の型定義."""

from typing import TypedDict


class AnalyzerConfigKwargs(TypedDict, total=False):
    """AnalyzerConfig.from_cli_args の引数型."""

    max_dim: int
    max_memory_mb: int
    min_chunk_size: int
    brightness_penalty_threshold: float
    brightness_penalty_value: float
    semantic_weight: float
    score_multiplier: float
    result_max_workers: int | None
    io_max_workers: int | None
