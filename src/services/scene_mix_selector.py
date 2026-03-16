"""scene mix と全体多様性を両立する選定ロジック."""

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..constants.scene_label import SceneLabel
from ..models.scored_candidate import ScoredCandidate
from ..models.selection_config import SelectionConfig
from ..utils.vector_utils import VectorUtils


@dataclass
class BucketPlan:
    """カテゴリ別の選定準備結果."""

    ordered_candidates: list[ScoredCandidate]
    leftovers: list[ScoredCandidate]


class SceneMixSelector:
    """画面種別ミックスを維持しつつ全体で多様な候補を選ぶ."""

    SCENE_ORDER = (SceneLabel.PLAY, SceneLabel.EVENT)
    BAND_LABELS = {
        1: ("mid",),
        2: ("low", "high"),
        3: ("low", "mid", "high"),
        4: ("low", "mid_low", "mid_high", "high"),
        5: ("low", "mid_low", "mid", "mid_high", "high"),
    }
    MIN_OUTLIER_SAMPLES = 4

    def __init__(self, config: SelectionConfig):
        """SceneMixSelectorを初期化する."""
        self.config = config

    def select(
        self,
        candidates: list[ScoredCandidate],
        num: int,
    ) -> tuple[list[ScoredCandidate], int, dict[str, int], dict[str, int]]:
        """比率に従って候補を選択する."""
        if num <= 0 or not candidates:
            empty_counts = {label.value: 0 for label in self.SCENE_ORDER}
            return [], 0, empty_counts, empty_counts

        targets = self._calculate_targets(num)
        scene_buckets = {
            label: [
                candidate
                for candidate in candidates
                if candidate.scene_assessment.scene_label == label
            ]
            for label in self.SCENE_ORDER
        }
        bucket_plans = {
            label: self._prepare_bucket(scene_buckets[label], targets[label])
            for label in self.SCENE_ORDER
        }

        selected: list[ScoredCandidate] = []
        selected_ids: set[int] = set()
        selected_features: list[np.ndarray[Any, Any]] = []
        rejected_by_similarity_ids: set[int] = set()

        for label in self.SCENE_ORDER:
            bucket_selected, bucket_rejected_ids, leftovers = self._select_from_stream(
                ordered_candidates=bucket_plans[label].ordered_candidates,
                target=targets[label],
                selected_ids=selected_ids,
                selected_features=selected_features,
            )
            bucket_plans[label].leftovers = leftovers + bucket_plans[label].leftovers
            self._extend_selection(
                selected=selected,
                selected_ids=selected_ids,
                selected_features=selected_features,
                additions=bucket_selected,
            )
            rejected_by_similarity_ids.update(bucket_rejected_ids)

        remaining_slots = num - len(selected)
        if remaining_slots > 0:
            fallback_stream = self._build_fallback_stream(
                [bucket_plans[label].leftovers for label in self.SCENE_ORDER]
            )
            fallback_selected, fallback_rejected_ids, _ = self._select_from_stream(
                ordered_candidates=fallback_stream,
                target=remaining_slots,
                selected_ids=selected_ids,
                selected_features=selected_features,
            )
            self._extend_selection(
                selected=selected,
                selected_ids=selected_ids,
                selected_features=selected_features,
                additions=fallback_selected,
            )
            rejected_by_similarity_ids.update(fallback_rejected_ids)

        actuals = {
            label.value: sum(
                1
                for candidate in selected
                if candidate.scene_assessment.scene_label == label
            )
            for label in self.SCENE_ORDER
        }
        target_map = {label.value: targets[label] for label in self.SCENE_ORDER}
        rejected_by_similarity = len(rejected_by_similarity_ids - selected_ids)
        return selected[:num], rejected_by_similarity, target_map, actuals

    def _calculate_targets(self, num: int) -> dict[SceneLabel, int]:
        """scene mix 比率から目標枚数を計算する."""
        raw_play = num * self.config.scene_mix.play
        raw_event = num * self.config.scene_mix.event
        play_target = int(raw_play)
        event_target = int(raw_event)
        remainder = num - (play_target + event_target)
        if remainder > 0 and raw_play - play_target >= raw_event - event_target:
            play_target += 1
        elif remainder > 0:
            event_target += 1
        return {
            SceneLabel.PLAY: play_target,
            SceneLabel.EVENT: event_target,
        }

    def _prepare_bucket(
        self,
        candidates: list[ScoredCandidate],
        target: int,
    ) -> BucketPlan:
        """カテゴリ別の候補を band 分散選定向けに並べ替える."""
        if not candidates:
            return BucketPlan([], [])

        for candidate in candidates:
            candidate.outlier_rejected = False
            candidate.score_band = None

        eligible_candidates = self._exclude_outliers(candidates)
        band_count = min(max(target, 1), 5)
        band_queues = self._build_band_queues(eligible_candidates, band_count)
        ordered_candidates = self._round_robin_bands(band_queues)
        return BucketPlan(ordered_candidates, [])

    def _exclude_outliers(
        self,
        candidates: list[ScoredCandidate],
    ) -> list[ScoredCandidate]:
        """selection_score の外れ値を除外する."""
        if len(candidates) < self.MIN_OUTLIER_SAMPLES:
            return list(candidates)

        scores = np.asarray(
            [candidate.selection_score for candidate in candidates],
            dtype=np.float32,
        )
        q1, q3 = np.percentile(scores, [25, 75])
        iqr = q3 - q1
        if np.isclose(iqr, 0.0):
            return list(candidates)

        lower = float(q1 - 1.5 * iqr)
        upper = float(q3 + 1.5 * iqr)
        eligible_candidates: list[ScoredCandidate] = []
        for candidate in candidates:
            score = candidate.selection_score
            if lower <= score <= upper:
                eligible_candidates.append(candidate)
            else:
                candidate.outlier_rejected = True
                candidate.score_band = "outlier"
        return eligible_candidates

    def _build_band_queues(
        self,
        candidates: list[ScoredCandidate],
        band_count: int,
    ) -> list[deque[ScoredCandidate]]:
        """候補を score band ごとのキューに分ける."""
        if not candidates:
            return []

        ordered = sorted(candidates, key=lambda item: item.selection_score)
        groups = np.array_split(np.asarray(ordered, dtype=object), band_count)
        band_names = self.BAND_LABELS[band_count]
        band_queues: list[deque[ScoredCandidate]] = []
        for band_name, group in zip(band_names, groups, strict=True):
            group_candidates = [candidate for candidate in group.tolist() if candidate is not None]
            if not group_candidates:
                continue
            band_center = float(
                np.mean([candidate.selection_score for candidate in group_candidates])
            )
            ordered_band = sorted(
                group_candidates,
                key=lambda candidate: (
                    -candidate.quality_score,
                    abs(candidate.selection_score - band_center),
                    candidate.path,
                ),
            )
            for candidate in ordered_band:
                candidate.score_band = band_name
            band_queues.append(deque(ordered_band))
        return band_queues

    @staticmethod
    def _round_robin_bands(
        band_queues: list[deque[ScoredCandidate]],
    ) -> list[ScoredCandidate]:
        """低 band から高 band へ均等に辿る順序を作る."""
        if not band_queues:
            return []

        ordered: list[ScoredCandidate] = []
        queues = [deque(queue) for queue in band_queues]
        while any(queues):
            for queue in queues:
                if queue:
                    ordered.append(queue.popleft())
        return ordered

    def _build_fallback_stream(
        self,
        leftovers_by_label: list[list[ScoredCandidate]],
    ) -> list[ScoredCandidate]:
        """不足分を補うための共通ストリームを組み立てる."""
        streams = [deque(candidates) for candidates in leftovers_by_label if candidates]
        ordered: list[ScoredCandidate] = []
        while any(streams):
            for stream in streams:
                if stream:
                    ordered.append(stream.popleft())
        return ordered

    def _select_from_stream(
        self,
        ordered_candidates: list[ScoredCandidate],
        target: int,
        selected_ids: set[int],
        selected_features: list[np.ndarray[Any, Any]],
    ) -> tuple[list[ScoredCandidate], set[int], list[ScoredCandidate]]:
        """順序付け済みストリームから類似度を見ながら採用する."""
        if target <= 0 or not ordered_candidates:
            return [], set(), list(ordered_candidates)

        selectable_candidates = [
            candidate for candidate in ordered_candidates if id(candidate) not in selected_ids
        ]
        if not selectable_candidates:
            return [], set(), []

        selected_indices, rejected_indices = VectorUtils.filter_by_similarity(
            candidates=[candidate.combined_features for candidate in selectable_candidates],
            num=target,
            similarity_threshold=self.config.similarity_threshold,
            compute_threshold_steps=self.config.compute_threshold_steps,
            seed_features=selected_features,
        )
        selected = [selectable_candidates[index] for index in selected_indices]
        rejected_by_similarity_ids = {
            id(selectable_candidates[index]) for index in rejected_indices
        }
        leftover_indices = (
            set(range(len(selectable_candidates))) - set(selected_indices) - rejected_indices
        )
        leftovers = [selectable_candidates[index] for index in sorted(leftover_indices)]
        return selected, rejected_by_similarity_ids, leftovers

    @staticmethod
    def _extend_selection(
        selected: list[ScoredCandidate],
        selected_ids: set[int],
        selected_features: list[np.ndarray[Any, Any]],
        additions: list[ScoredCandidate],
    ) -> None:
        """選択結果と全体類似度チェック用の状態をまとめて更新する."""
        selected.extend(additions)
        selected_ids.update(id(candidate) for candidate in additions)
        selected_features.extend(candidate.combined_features for candidate in additions)
