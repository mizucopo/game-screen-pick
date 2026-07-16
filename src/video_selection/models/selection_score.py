"""一回のgreedy選択時点におけるutility内訳。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SelectionScore:
    """selector判断を再現する数値componentを保持する。"""

    base_utility: float
    spoiler_penalty: float
    coverage_bonus: float
    temporal_diversity_penalty: float
    marginal_utility: float
    similarity_pass: float
    nearest_selected_similarity: float | None
