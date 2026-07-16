"""Video Set最終選定の結果。"""

from dataclasses import dataclass

from .rejected_blog_candidate import RejectedBlogCandidate
from .selected_blog_image import SelectedBlogImage


@dataclass(frozen=True)
class VideoSetSelectionResult:
    """選択画像と選定時のcoverage・similarity状態を保持する。"""

    selected: tuple[SelectedBlogImage, ...]
    rejected: tuple[RejectedBlogCandidate, ...]
    requested_count: int
    blog_image_type_targets: dict[str, int]
    blog_image_type_actuals: dict[str, int]
    final_similarity_ceiling: float
    major_spoiler_limit: int | None
    annotated_candidate_count: int
    shortlist_expansion_count: int
    all_candidate_moments_exhausted: bool

    @property
    def shortfall(self) -> bool:
        """要求枚数へ到達しなかったかを返す。"""
        return len(self.selected) < self.requested_count

    @property
    def major_spoiler_selected_count(self) -> int:
        """Major Spoiler Signalを持つ選択画像数を返す。"""
        return sum(
            item.candidate.annotation.spoiler_risk == "high" for item in self.selected
        )

    @property
    def rejection_counts(self) -> dict[str, int]:
        """未採用候補をstable reason code別に集計する。"""
        counts: dict[str, int] = {}
        for item in self.rejected:
            counts[item.reason_code.value] = counts.get(item.reason_code.value, 0) + 1
        return counts
