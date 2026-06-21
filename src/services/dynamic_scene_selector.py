"""動的scene catalog向けの選定ロジック."""

import math
from collections import defaultdict, deque
from collections.abc import Sequence

import numpy as np

from ..models.scene_selection_role import SceneSelectionRole
from ..models.scored_candidate import ScoredCandidate
from ..models.selection_annotation import SelectionAnnotation
from ..models.selection_result import SelectionResult
from ..utils.vector_utils import VectorUtils
from .variant_group_assigner import VariantGroupAssigner


class DynamicSceneSelector:
    """動的sceneを均等に扱いながら候補を選ぶ."""

    RECURRING_GAMEPLAY_SIMILARITY_THRESHOLD = 0.98

    def __init__(
        self,
        similarity_threshold: float,
        threshold_steps: list[float],
        variant_similarity_threshold: float = 0.95,
    ) -> None:
        """selectorを初期化する."""
        self.similarity_threshold = similarity_threshold
        self.threshold_steps = threshold_steps
        self._variant_group_assigner = VariantGroupAssigner(
            variant_similarity_threshold
        )

    def select(
        self,
        candidates: Sequence[ScoredCandidate],
        num: int,
    ) -> SelectionResult[ScoredCandidate]:
        """sceneごとに自動均等配分して候補を選ぶ."""
        if num <= 0 or not candidates:
            return SelectionResult([], 0, {}, {})

        scene_order = self._scene_order(candidates)
        scene_roles = self._scene_roles(scene_order, candidates)
        targets = self._calculate_targets(scene_order, num, candidates, scene_roles)
        annotations_by_path: dict[str, SelectionAnnotation] = {}
        variant_groups_by_path = self._variant_group_assigner.assign(candidates)
        streams = self._build_scene_streams(
            candidates,
            scene_order,
            annotations_by_path,
            variant_groups_by_path,
        )
        ordered_candidates, cinematic_overflow_start = self._round_robin_scene_streams(
            streams,
            targets,
            scene_roles,
        )
        selected, rejected_by_similarity = self._select_with_similarity(
            ordered_candidates,
            num,
            cinematic_overflow_start,
        )
        actuals = {
            scene: sum(1 for candidate in selected if candidate.scene_slug == scene)
            for scene in scene_order
        }
        return SelectionResult(
            selected=selected,
            rejected_by_similarity=rejected_by_similarity,
            target_counts=targets,
            actual_counts=actuals,
            annotations_by_path=annotations_by_path,
        )

    @staticmethod
    def _scene_order(candidates: Sequence[ScoredCandidate]) -> list[str]:
        """候補に現れた順序でscene slug一覧を返す."""
        seen: set[str] = set()
        result: list[str] = []
        for candidate in candidates:
            if candidate.scene_slug not in seen:
                seen.add(candidate.scene_slug)
                result.append(candidate.scene_slug)
        return result

    @staticmethod
    def _calculate_targets(
        scene_order: list[str],
        num: int,
        candidates: Sequence[ScoredCandidate],
        scene_roles: dict[str, SceneSelectionRole],
    ) -> dict[str, int]:
        """sceneごとの自動均等目標枚数を計算する."""
        scene_counts = {
            scene: sum(1 for candidate in candidates if candidate.scene_slug == scene)
            for scene in scene_order
        }
        allocation_order = DynamicSceneSelector._target_allocation_order(
            scene_order,
            num,
            candidates,
        )
        remaining = min(num, len(candidates))
        cinematic_limit = DynamicSceneSelector._cinematic_target_limit(
            requested_count=remaining,
            scene_counts=scene_counts,
            scene_roles=scene_roles,
        )
        cinematic_used = 0
        targets = dict.fromkeys(scene_order, 0)
        while remaining > 0:
            progressed = False
            for scene in allocation_order:
                if targets[scene] >= scene_counts[scene]:
                    continue
                if (
                    scene_roles[scene] == SceneSelectionRole.CINEMATIC
                    and cinematic_used >= cinematic_limit
                ):
                    continue
                targets[scene] += 1
                if scene_roles[scene] == SceneSelectionRole.CINEMATIC:
                    cinematic_used += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
            if not progressed:
                break
        return targets

    @staticmethod
    def _scene_roles(
        scene_order: list[str],
        candidates: Sequence[ScoredCandidate],
    ) -> dict[str, SceneSelectionRole]:
        """sceneごとのselection roleを返す."""
        roles = dict.fromkeys(scene_order, SceneSelectionRole.ORDINARY)
        for candidate in candidates:
            if roles[candidate.scene_slug] == SceneSelectionRole.ORDINARY:
                roles[candidate.scene_slug] = candidate.scene_selection_role
        return roles

    @staticmethod
    def _cinematic_target_limit(
        *,
        requested_count: int,
        scene_counts: dict[str, int],
        scene_roles: dict[str, SceneSelectionRole],
    ) -> int:
        """cinematic scene全体のsoft cap枚数を返す."""
        total_cinematic = sum(
            count
            for scene, count in scene_counts.items()
            if scene_roles[scene] == SceneSelectionRole.CINEMATIC
        )
        if requested_count <= 0 or total_cinematic == 0:
            return 0
        non_cinematic_capacity = sum(
            count
            for scene, count in scene_counts.items()
            if scene_roles[scene] != SceneSelectionRole.CINEMATIC
        )
        soft_cap = max(1, math.ceil(requested_count * 0.1))
        overflow_needed = max(0, requested_count - non_cinematic_capacity)
        return min(total_cinematic, max(soft_cap, overflow_needed))

    @staticmethod
    def _target_allocation_order(
        scene_order: list[str],
        num: int,
        candidates: Sequence[ScoredCandidate],
    ) -> list[str]:
        """枠がscene数より少ない場合はsceneの最高score順で割り当てる."""
        if num >= len(scene_order):
            return scene_order

        scene_index = {scene: index for index, scene in enumerate(scene_order)}
        best_score_by_scene = DynamicSceneSelector._best_score_by_scene(scene_order)
        for candidate in candidates:
            candidate_score = (candidate.selection_score, candidate.quality_score)
            current_score = best_score_by_scene[candidate.scene_slug]
            if candidate_score > current_score:
                best_score_by_scene[candidate.scene_slug] = candidate_score

        return sorted(
            scene_order,
            key=lambda scene: (
                -best_score_by_scene[scene][0],
                -best_score_by_scene[scene][1],
                scene_index[scene],
            ),
        )

    @staticmethod
    def _best_score_by_scene(scene_order: list[str]) -> dict[str, tuple[float, float]]:
        """sceneごとの最高score初期値を返す."""
        return dict.fromkeys(scene_order, (0.0, 0.0))

    @staticmethod
    def _build_scene_streams(
        candidates: Sequence[ScoredCandidate],
        scene_order: list[str],
        annotations_by_path: dict[str, SelectionAnnotation],
        variant_groups_by_path: dict[str, str],
    ) -> dict[str, deque[ScoredCandidate]]:
        """sceneごとの候補streamを作る."""
        grouped: dict[str, list[ScoredCandidate]] = defaultdict(list)
        for candidate in candidates:
            grouped[candidate.scene_slug].append(candidate)
        streams: dict[str, deque[ScoredCandidate]] = {}
        for scene in scene_order:
            ordered_by_score = sorted(
                grouped[scene],
                key=lambda candidate: (
                    -candidate.selection_score,
                    -candidate.quality_score,
                    candidate.path,
                ),
            )
            ordered = DynamicSceneSelector._round_robin_variant_groups(
                ordered_by_score,
                variant_groups_by_path,
            )
            for candidate in ordered:
                annotations_by_path[candidate.path] = SelectionAnnotation(
                    score_band="scene",
                    outlier_rejected=False,
                    variant_group=variant_groups_by_path[candidate.path],
                )
            streams[scene] = deque(ordered)
        return streams

    @staticmethod
    def _round_robin_variant_groups(
        ordered_candidates: list[ScoredCandidate],
        variant_groups_by_path: dict[str, str],
    ) -> list[ScoredCandidate]:
        """variant groupごとに1枚目を優先する順序へ並べ替える."""
        grouped: dict[str, deque[ScoredCandidate]] = defaultdict(deque)
        group_order: list[str] = []
        for candidate in ordered_candidates:
            group_id = variant_groups_by_path[candidate.path]
            if group_id not in grouped:
                group_order.append(group_id)
            grouped[group_id].append(candidate)

        result: list[ScoredCandidate] = []
        while any(grouped.values()):
            for group_id in group_order:
                if grouped[group_id]:
                    result.append(grouped[group_id].popleft())
        return result

    @staticmethod
    def _round_robin_scene_streams(
        streams: dict[str, deque[ScoredCandidate]],
        targets: dict[str, int],
        scene_roles: dict[str, SceneSelectionRole],
    ) -> tuple[list[ScoredCandidate], int]:
        """scene streamを目標枚数までround-robinで並べる."""
        selected_order: list[ScoredCandidate] = []
        cinematic_overflow_start: int | None = None
        used = dict.fromkeys(targets, 0)
        while any(streams.values()):
            progressed = False
            for scene, stream in streams.items():
                if not stream:
                    continue
                if used[scene] >= targets[scene]:
                    continue
                selected_order.append(stream.popleft())
                used[scene] += 1
                progressed = True
            if not progressed:
                cinematic_overflow_start = (
                    DynamicSceneSelector._append_remaining_streams(
                        selected_order,
                        streams,
                        scene_roles,
                    )
                )
        if cinematic_overflow_start is None:
            cinematic_overflow_start = len(selected_order)
        return selected_order, cinematic_overflow_start

    @staticmethod
    def _append_remaining_streams(
        selected_order: list[ScoredCandidate],
        streams: dict[str, deque[ScoredCandidate]],
        scene_roles: dict[str, SceneSelectionRole],
    ) -> int:
        """目標超過候補をnon-cinematic優先で末尾に追加する."""
        for scene, stream in streams.items():
            if scene_roles[scene] == SceneSelectionRole.CINEMATIC:
                continue
            selected_order.extend(stream)
            stream.clear()
        cinematic_overflow_start = len(selected_order)
        for scene, stream in streams.items():
            if scene_roles[scene] != SceneSelectionRole.CINEMATIC:
                continue
            selected_order.extend(stream)
            stream.clear()
        return cinematic_overflow_start

    def _select_with_similarity(
        self,
        ordered_candidates: list[ScoredCandidate],
        num: int,
        cinematic_overflow_start: int,
    ) -> tuple[list[ScoredCandidate], int]:
        """類似度を見ながら候補を採用する."""
        target_count = min(num, len(ordered_candidates))
        selected_indices, rejected_indices = self._select_indices_with_role_similarity(
            ordered_candidates,
            target_count,
            candidate_indices=range(cinematic_overflow_start),
        )
        if len(selected_indices) < target_count:
            selected_indices, overflow_rejected_indices = (
                self._select_indices_with_role_similarity(
                    ordered_candidates,
                    target_count,
                    candidate_indices=range(
                        cinematic_overflow_start,
                        len(ordered_candidates),
                    ),
                    seed_indices=selected_indices,
                )
            )
            rejected_indices |= overflow_rejected_indices
        selected = [ordered_candidates[index] for index in selected_indices]
        return selected, len(rejected_indices)

    def _select_indices_with_role_similarity(
        self,
        ordered_candidates: list[ScoredCandidate],
        num: int,
        candidate_indices: Sequence[int] | None = None,
        seed_indices: Sequence[int] | None = None,
    ) -> tuple[list[int], set[int]]:
        """roleに応じた類似度しきい値で候補indexを選ぶ."""
        if num <= 0 or not ordered_candidates:
            return [], set()

        if candidate_indices is None:
            candidate_indices = range(len(ordered_candidates))
        if seed_indices is None:
            seed_indices = []
        normalized_features = VectorUtils.normalize_feature_vectors(
            [candidate.combined_features for candidate in ordered_candidates]
        )
        target_count = min(num, len(ordered_candidates))
        feature_dim = len(normalized_features[0])
        selected_features_matrix = np.zeros(
            (target_count, feature_dim),
            dtype=np.float32,
        )
        selected_indices = list(seed_indices)
        selected_index_set = set(selected_indices)
        rejected_by_similarity_set: set[int] = set()
        recurring_gameplay_threshold = self.RECURRING_GAMEPLAY_SIMILARITY_THRESHOLD
        selected_count = 0
        for seed_index in selected_indices:
            selected_features_matrix[selected_count] = normalized_features[seed_index]
            selected_count += 1

        for threshold in self.threshold_steps:
            for index in candidate_indices:
                if index in selected_index_set:
                    continue
                if len(selected_indices) >= target_count:
                    break

                candidate = ordered_candidates[index]
                feature = normalized_features[index]
                candidate_threshold = self._candidate_similarity_threshold(
                    candidate,
                    threshold,
                    recurring_gameplay_threshold,
                )
                similarities = selected_features_matrix[:selected_count] @ feature
                is_similar = bool(np.any(similarities > candidate_threshold))
                if is_similar:
                    rejected_by_similarity_set.add(index)
                    continue

                selected_indices.append(index)
                selected_index_set.add(index)
                selected_features_matrix[selected_count] = feature
                selected_count += 1

            if len(selected_indices) >= target_count:
                break

        return selected_indices, rejected_by_similarity_set - selected_index_set

    @staticmethod
    def _candidate_similarity_threshold(
        candidate: ScoredCandidate,
        threshold: float,
        recurring_gameplay_threshold: float,
    ) -> float:
        """候補roleに応じた類似度しきい値を返す."""
        if candidate.scene_selection_role != SceneSelectionRole.RECURRING_GAMEPLAY:
            return threshold
        return max(threshold, recurring_gameplay_threshold)
