"""scene mix と activity mix を両立する選定ロジック."""

from ..constants.scene_label import SceneLabel
from ..models.scored_candidate import ScoredCandidate
from ..models.selection_config import SelectionConfig
from ..models.selection_profile import SelectionProfile
from ..utils.vector_utils import VectorUtils


class SceneMixSelector:
    """画面種別ミックスを維持しつつ候補を選ぶ.

    まず gameplay / event / other の比率を守り、
    その後で各scene bucket内の活動量偏りを緩和する二段構成を取る。
    """

    SCENE_ORDER = (SceneLabel.GAMEPLAY, SceneLabel.EVENT, SceneLabel.OTHER)

    def __init__(self, config: SelectionConfig):
        """SceneMixSelectorを初期化する.

        Args:
            config: 類似度しきい値、scene mix比率などを含む選択設定。
        """
        self.config = config

    def select(
        self,
        candidates: list[ScoredCandidate],
        num: int,
        profile: SelectionProfile,
    ) -> tuple[list[ScoredCandidate], int, dict[str, int], dict[str, int]]:
        """比率に従って候補を選択する.

        処理順は以下の通り。
        1. `selection_score` 順に並べた候補を scene label ごとに分桶する
        2. scene mix比率から目標枚数を求める
        3. 各bucket内で類似度フィルタを通し、多様プールを作る
        4. 多様プール内で activity mix を適用し、低・中・高活動量の偏りを減らす
        5. 不足枠があれば `gameplay -> event -> other` の順で再配分する

        `other` は少量含める前提のため、最初の目標枠を超えて
        積極的に増やすのではなく、再配分時の最後の受け皿として扱う。

        Args:
            candidates: 画面種別と選定スコアが付与済みの候補。
            num: 最終的に選びたい枚数。
            profile: activity mix比率を持つ解決済みプロファイル。

        Returns:
            1. 選択された候補
            2. 類似度フィルタで落ちた件数
            3. scene labelごとの目標枚数
            4. 実際に選ばれたscene labelごとの枚数
        """
        if not candidates:
            empty_counts = {label.value: 0 for label in self.SCENE_ORDER}
            return [], 0, empty_counts, empty_counts

        sorted_candidates = sorted(
            candidates, key=lambda item: item.selection_score, reverse=True
        )
        targets = self._calculate_targets(num)
        scene_buckets = {
            label: [
                candidate
                for candidate in sorted_candidates
                if candidate.scene_assessment.scene_label == label
            ]
            for label in self.SCENE_ORDER
        }

        selected: list[ScoredCandidate] = []
        selected_ids: set[int] = set()
        rejected_by_similarity = 0

        for label in self.SCENE_ORDER:
            bucket_selected, bucket_rejected = self._select_scene_bucket(
                candidates=scene_buckets[label],
                target=targets[label],
                profile=profile,
                selected_ids=selected_ids,
            )
            selected.extend(bucket_selected)
            selected_ids.update(id(candidate) for candidate in bucket_selected)
            rejected_by_similarity += bucket_rejected

        remaining_slots = num - len(selected)
        if remaining_slots > 0:
            for label in self.SCENE_ORDER:
                if remaining_slots <= 0:
                    break
                remaining_candidates = [
                    candidate
                    for candidate in scene_buckets[label]
                    if id(candidate) not in selected_ids
                ]
                bucket_selected, bucket_rejected = self._select_scene_bucket(
                    candidates=remaining_candidates,
                    target=remaining_slots,
                    profile=profile,
                    selected_ids=selected_ids,
                )
                selected.extend(bucket_selected)
                selected_ids.update(id(candidate) for candidate in bucket_selected)
                rejected_by_similarity += bucket_rejected
                remaining_slots = num - len(selected)

        selected.sort(key=lambda item: item.selection_score, reverse=True)
        actuals = {
            label.value: sum(
                1
                for candidate in selected
                if candidate.scene_assessment.scene_label == label
            )
            for label in self.SCENE_ORDER
        }
        target_map = {label.value: targets[label] for label in self.SCENE_ORDER}
        return selected[:num], rejected_by_similarity, target_map, actuals

    def _calculate_targets(self, num: int) -> dict[SceneLabel, int]:
        """scene mix 比率から目標枚数を計算する.

        小数点以下はまず切り捨て、余りは端数の大きいbucketから順に配る。
        これにより合計枚数を保ったまま比率へ最も近い整数配分を作る。

        Args:
            num: 選択したい総枚数。

        Returns:
            scene labelごとの目標枚数。
        """
        ratios = {
            SceneLabel.GAMEPLAY: self.config.scene_mix.gameplay,
            SceneLabel.EVENT: self.config.scene_mix.event,
            SceneLabel.OTHER: self.config.scene_mix.other,
        }
        raw_targets = {label: num * ratio for label, ratio in ratios.items()}
        base_targets = {label: int(value) for label, value in raw_targets.items()}
        remainder = num - sum(base_targets.values())
        remainders = sorted(
            raw_targets.items(),
            key=lambda item: item[1] - int(item[1]),
            reverse=True,
        )
        for index, (label, _) in enumerate(remainders):
            if index < remainder:
                base_targets[label] += 1
        return base_targets

    def _select_scene_bucket(
        self,
        candidates: list[ScoredCandidate],
        target: int,
        profile: SelectionProfile,
        selected_ids: set[int],
    ) -> tuple[list[ScoredCandidate], int]:
        """単一scene bucketから候補を選択する.

        まず未選択候補だけに絞り、類似度フィルタで多様プールを作る。
        その後、同じscene bucket内で `activity_mix` を適用して
        動きの少ない画像と多い画像の偏りを抑えながら最終採用数を決める。

        Args:
            candidates: ひとつのscene labelに属する候補群。
            target: このbucketから確保したい枚数。
            profile: 活動量配分を決める解決済みプロファイル。
            selected_ids: 他bucketで既に選ばれた候補のID集合。

        Returns:
            1. このbucketから選ばれた候補
            2. 類似度フィルタで除外された件数
        """
        if target <= 0 or not candidates:
            return [], 0

        unselected_candidates = [
            candidate for candidate in candidates if id(candidate) not in selected_ids
        ]
        if not unselected_candidates:
            return [], 0

        pool_size = min(len(unselected_candidates), max(target * 3, target))
        selected_indices, rejected_by_similarity = VectorUtils.filter_by_similarity(
            candidates=[
                candidate.combined_features for candidate in unselected_candidates
            ],
            num=pool_size,
            similarity_threshold=self.config.similarity_threshold,
            compute_threshold_steps=self.config.compute_threshold_steps,
        )
        diverse_pool = [unselected_candidates[idx] for idx in sorted(selected_indices)]
        if len(diverse_pool) < target:
            seen_ids = {id(candidate) for candidate in diverse_pool}
            for candidate in unselected_candidates:
                if id(candidate) not in seen_ids:
                    diverse_pool.append(candidate)
                    seen_ids.add(id(candidate))
                if len(diverse_pool) >= target:
                    break

        selected = self._select_with_activity_mix(diverse_pool, target, profile)
        return selected, rejected_by_similarity

    @staticmethod
    def _select_with_activity_mix(
        candidates: list[ScoredCandidate],
        target: int,
        profile: SelectionProfile,
    ) -> list[ScoredCandidate]:
        """activity mix を使って偏りを減らす.

        候補を `activity_score` 順に並べて低・中・高の3帯域へ分け、
        プロファイルで定義された比率に従って採用数を割り当てる。
        各帯域で枠が余った場合は、残り候補を `selection_score` 順で補充する。

        Args:
            candidates: 類似度フィルタ後の多様プール。
            target: このscene bucketから最終的に取りたい枚数。
            profile: 帯域ごとの活動量比率を持つプロファイル。

        Returns:
            活動量バランスを取った候補リスト。
        """
        if len(candidates) <= target:
            return sorted(
                candidates, key=lambda item: item.selection_score, reverse=True
            )

        ranked = sorted(candidates, key=lambda item: item.activity_score)
        low_end = len(ranked) // 3
        high_end = (len(ranked) * 2) // 3
        buckets = {
            "low": ranked[:low_end],
            "mid": ranked[low_end:high_end],
            "high": ranked[high_end:],
        }
        ratios = {
            "low": profile.activity_mix_ratio[0],
            "mid": profile.activity_mix_ratio[1],
            "high": profile.activity_mix_ratio[2],
        }
        raw_targets = {name: target * ratio for name, ratio in ratios.items()}
        bucket_targets = {name: int(value) for name, value in raw_targets.items()}
        remainder = target - sum(bucket_targets.values())
        ordered_remainders = sorted(
            raw_targets.items(),
            key=lambda item: item[1] - int(item[1]),
            reverse=True,
        )
        for index, (name, _) in enumerate(ordered_remainders):
            if index < remainder:
                bucket_targets[name] += 1

        selected: list[ScoredCandidate] = []
        for name in ("low", "mid", "high"):
            bucket_candidates = sorted(
                buckets[name],
                key=lambda item: item.selection_score,
                reverse=True,
            )
            selected.extend(bucket_candidates[: bucket_targets[name]])

        if len(selected) < target:
            selected_ids = {id(candidate) for candidate in selected}
            leftovers = sorted(
                [
                    candidate
                    for candidate in candidates
                    if id(candidate) not in selected_ids
                ],
                key=lambda item: item.selection_score,
                reverse=True,
            )
            selected.extend(leftovers[: target - len(selected)])

        selected.sort(key=lambda item: item.selection_score, reverse=True)
        return selected[:target]
