"""カテゴリ別の選定準備結果."""

from dataclasses import dataclass

from .scored_candidate import ScoredCandidate


@dataclass
class BucketPlan:
    """カテゴリ別の選定準備結果."""

    ordered_candidates: list[ScoredCandidate]
    leftovers: list[ScoredCandidate]
