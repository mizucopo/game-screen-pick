"""Video Set selectorのstable rejection reason。"""

from enum import StrEnum


class SelectionRejectionReason(StrEnum):
    """未採用Blog Candidateの排他的な主理由。"""

    TITLE_LIMIT = "title_limit"
    VISUAL_NEAR_DUPLICATE = "visual_near_duplicate"
    SIMILARITY_CEILING = "similarity_ceiling"
    SPOILER_MONOTONICITY_GUARD = "spoiler_monotonicity_guard"
    LOWER_MARGINAL_UTILITY = "lower_marginal_utility"
