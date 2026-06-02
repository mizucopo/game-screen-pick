"""カテゴリ別の選定準備結果."""

from dataclasses import dataclass
from typing import Generic, TypeVar

from .scene_mix_candidate import SceneMixCandidate

SceneMixCandidateT = TypeVar("SceneMixCandidateT", bound=SceneMixCandidate)


@dataclass
class BucketPlan(Generic[SceneMixCandidateT]):
    """カテゴリ別の選定準備結果."""

    ordered_candidates: list[SceneMixCandidateT]
    leftovers: list[SceneMixCandidateT]
