"""Video Set selectorが採用しなかったBlog Candidate。"""

from dataclasses import dataclass

from .blog_candidate import BlogCandidate
from .selection_rejection_reason import SelectionRejectionReason
from .selection_score import SelectionScore


@dataclass(frozen=True)
class RejectedBlogCandidate:
    """stable reasonと採用を妨げた選択画像を保持する。"""

    candidate: BlogCandidate
    reason_code: SelectionRejectionReason
    counterfactual_score: SelectionScore
    blocked_by_image_id: str | None
    nearest_selected_image_id: str | None
    similarity: float | None
    variant_group_id: str
