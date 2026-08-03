"""決定的selectorが選んだ一つのBlog Candidate。"""

from dataclasses import dataclass

from .blog_candidate import BlogCandidate
from .selection_score import SelectionScore
from .semantic_duplicate_basis import SemanticDuplicateBasis


@dataclass(frozen=True)
class SelectedBlogImage:
    """選択順、数値内訳、stable reason codeを保持する。"""

    candidate: BlogCandidate
    selection_index: int
    score: SelectionScore
    reason_codes: tuple[str, ...]
    variant_group_id: str
    tie_break_applied: bool
    semantic_group_id: str | None = None
    semantic_group_basis: SemanticDuplicateBasis | None = None
