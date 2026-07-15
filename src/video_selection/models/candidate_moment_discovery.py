"""density適用後のCandidate Moment discovery。"""

from dataclasses import dataclass

from .candidate_moment import CandidateMoment


@dataclass(frozen=True)
class CandidateMomentDiscovery:
    """理論上限と実際に発見されたMomentを保持する。"""

    density_cap: int
    moments: tuple[CandidateMoment, ...]

    def __post_init__(self) -> None:
        """実Moment数がdensity上限を超えないことを検証する。"""
        if self.density_cap < 1 or len(self.moments) > self.density_cap:
            msg = "Candidate Moment数がdensity上限と一致しません"
            raise ValueError(msg)
