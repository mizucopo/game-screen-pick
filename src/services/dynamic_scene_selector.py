"""動的scene catalog向けの選定ロジック."""

from collections import defaultdict, deque
from collections.abc import Sequence

from ..models.scored_candidate import ScoredCandidate
from ..models.selection_annotation import SelectionAnnotation
from ..models.selection_result import SelectionResult
from ..utils.vector_utils import VectorUtils
from .variant_group_assigner import VariantGroupAssigner


class DynamicSceneSelector:
    """動的sceneを均等に扱いながら候補を選ぶ."""

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
        targets = self._calculate_targets(scene_order, num, candidates)
        annotations_by_path: dict[str, SelectionAnnotation] = {}
        variant_groups_by_path = self._variant_group_assigner.assign(candidates)
        streams = self._build_scene_streams(
            candidates,
            scene_order,
            annotations_by_path,
            variant_groups_by_path,
        )
        ordered_candidates = self._round_robin_scene_streams(streams, targets)
        selected, rejected_by_similarity = self._select_with_similarity(
            ordered_candidates,
            num,
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
        targets = dict.fromkeys(scene_order, 0)
        while remaining > 0:
            progressed = False
            for scene in allocation_order:
                if targets[scene] < scene_counts[scene]:
                    targets[scene] += 1
                    remaining -= 1
                    progressed = True
                    if remaining == 0:
                        break
            if not progressed:
                break
        return targets

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
        best_score_by_scene = {
            scene: max(
                (
                    (candidate.selection_score, candidate.quality_score)
                    for candidate in candidates
                    if candidate.scene_slug == scene
                ),
                default=(0.0, 0.0),
            )
            for scene in scene_order
        }
        return sorted(
            scene_order,
            key=lambda scene: (
                -best_score_by_scene[scene][0],
                -best_score_by_scene[scene][1],
                scene_index[scene],
            ),
        )

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
    ) -> list[ScoredCandidate]:
        """scene streamを目標枚数までround-robinで並べる."""
        selected_order: list[ScoredCandidate] = []
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
                for stream in streams.values():
                    selected_order.extend(stream)
                    stream.clear()
        return selected_order

    def _select_with_similarity(
        self,
        ordered_candidates: list[ScoredCandidate],
        num: int,
    ) -> tuple[list[ScoredCandidate], int]:
        """類似度を見ながら候補を採用する."""
        selected_indices, rejected_indices = VectorUtils.filter_by_similarity(
            candidates=[
                candidate.combined_features for candidate in ordered_candidates
            ],
            num=num,
            similarity_threshold=self.similarity_threshold,
            compute_threshold_steps=lambda _threshold: self.threshold_steps,
        )
        selected = [ordered_candidates[index] for index in selected_indices]
        return selected, len(rejected_indices)
