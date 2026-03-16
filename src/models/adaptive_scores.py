"""入力全体に対する相対スコア."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AdaptiveScores:
    """入力集合に対する相対的な内容スコア."""

    information_score: float
    distinctiveness_score: float
    visibility_score: float
