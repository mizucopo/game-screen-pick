"""Video Set最終選定の結果。"""

from dataclasses import dataclass

from .candidate_annotation import SELECTION_COVERAGE_FACETS
from .rejected_blog_candidate import RejectedBlogCandidate
from .selected_blog_image import SelectedBlogImage

CONDITIONAL_COVERAGE_MINIMUM_REQUEST_COUNT = 10


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

    @property
    def selection_coverage_eligible_counts(self) -> dict[str, int]:
        """説明価値を持つ条件付きcoverage候補を役割別に集計する。"""
        counts: dict[str, int] = dict.fromkeys(SELECTION_COVERAGE_FACETS, 0)
        for candidate in (
            *(item.candidate for item in self.selected),
            *(item.candidate for item in self.rejected),
        ):
            facet = candidate.annotation.selection_coverage_facet
            if facet is not None and candidate.annotation.explanation_value != "none":
                counts[facet] += 1
        return counts

    @property
    def selection_coverage_minimums(self) -> dict[str, int]:
        """要求枚数と有効候補に応じた条件付き最低数を返す。"""
        eligible = self.selection_coverage_eligible_counts
        applies = self.requested_count >= CONDITIONAL_COVERAGE_MINIMUM_REQUEST_COUNT
        return {
            facet: int(applies and eligible[facet] > 0)
            for facet in SELECTION_COVERAGE_FACETS
        }

    @property
    def selection_coverage_actuals(self) -> dict[str, int]:
        """選択済み画像の条件付きcoverageを役割別に集計する。"""
        counts: dict[str, int] = dict.fromkeys(SELECTION_COVERAGE_FACETS, 0)
        for item in self.selected:
            facet = item.candidate.annotation.selection_coverage_facet
            if facet is not None:
                counts[facet] += 1
        return counts

    @property
    def selection_coverage_reallocated(self) -> dict[str, bool]:
        """最低候補がないか制約優先で未充足となった枠の解放状態を返す。"""
        applies = self.requested_count >= CONDITIONAL_COVERAGE_MINIMUM_REQUEST_COUNT
        actuals = self.selection_coverage_actuals
        return {
            facet: applies and actuals[facet] == 0
            for facet in SELECTION_COVERAGE_FACETS
        }
