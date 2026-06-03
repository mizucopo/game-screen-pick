"""scene内の重複差分をvariant groupへまとめる."""

from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from ..models.scored_candidate import ScoredCandidate
from ..utils.vector_utils import VectorUtils


class VariantGroupAssigner:
    """候補をscene内のvariant groupへ割り当てる."""

    def __init__(self, similarity_threshold: float = 0.95) -> None:
        """assignerを初期化する."""
        self.similarity_threshold = similarity_threshold

    def assign(self, candidates: Sequence[ScoredCandidate]) -> dict[str, str]:
        """候補pathごとのvariant group idを返す."""
        grouped_by_scene: dict[str, list[ScoredCandidate]] = defaultdict(list)
        for candidate in candidates:
            grouped_by_scene[candidate.scene_slug].append(candidate)

        result: dict[str, str] = {}
        for scene_slug, scene_candidates in grouped_by_scene.items():
            representatives: list[np.ndarray] = []
            group_ids: list[str] = []
            for candidate in scene_candidates:
                normalized = VectorUtils.safe_l2_normalize(candidate.combined_features)
                group_id = self._find_group_id(normalized, representatives, group_ids)
                if group_id is None:
                    group_id = f"{scene_slug}_{len(representatives) + 1:03d}"
                    representatives.append(normalized)
                    group_ids.append(group_id)
                result[candidate.path] = group_id
        return result

    def _find_group_id(
        self,
        feature: np.ndarray,
        representatives: list[np.ndarray],
        group_ids: list[str],
    ) -> str | None:
        """既存代表に近いgroup idを探す."""
        for representative, group_id in zip(representatives, group_ids, strict=True):
            if float(representative @ feature) >= self.similarity_threshold:
                return group_id
        return None
