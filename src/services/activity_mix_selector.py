"""活動量ミックスを考慮した画像選択ロジック."""

import numpy as np

from ..models.activity_bucket import ActivityBucket
from ..models.bucketed_image import BucketedImage
from ..models.image_metrics import ImageMetrics
from ..models.selection_config import SelectionConfig


class ActivityMixSelector:
    """活動量ミックスを考慮した画像選択クラス.

    活動量（LOW/MID/HIGH）に基づいてバランスよく画像を選択し、
    類似度フィルタリングも適用する。
    """

    def __init__(
        self,
        activity_weights: dict[str, float],
        config: SelectionConfig,
    ):
        """セレクターを初期化する.

        Args:
            activity_weights: 活動量計算用の重み
            config: 選択設定
        """
        self.activity_weights = activity_weights
        self.config = config

    def select(
        self,
        all_results: list[ImageMetrics],
        num: int,
        similarity_threshold: float,
    ) -> tuple[list[ImageMetrics], int]:
        """活動量ミックスを考慮して画像を選択する.

        Args:
            all_results: 解析済みの画像メトリクスリスト（スコア降順ソート済み）
            num: 選択する画像数
            similarity_threshold: 類似度の閾値

        Returns:
            選択された画像メトリクスのリストと類似度で除外された数のタプル
        """
        if not all_results:
            return [], 0

        # 各画像に活動量スコアを計算してバケット分け
        bucketed_images = self._bucket_by_activity(all_results)

        # 活動量ミックスの比率を取得
        low_ratio, mid_ratio, high_ratio = self.config.activity_mix_ratio

        # 各バケットから選択する数を計算（小数を保持して後で調整）
        low_target = num * low_ratio
        mid_target = num * mid_ratio
        high_target = num * high_ratio

        # 小数部分を保持して、余りを配分
        targets = [
            (ActivityBucket.LOW, low_target),
            (ActivityBucket.MID, mid_target),
            (ActivityBucket.HIGH, high_target),
        ]
        # 小数部分の大きい順にソートして、余りを配分
        targets_sorted = sorted(targets, key=lambda x: x[1] - int(x[1]), reverse=True)

        # 基本の整数部分
        base_total = sum(int(t) for _, t in targets)
        remainder = num - base_total

        # 余りを小数部分の大きいバケットに配分
        bucket_targets: dict[ActivityBucket, int] = {}
        for i, (bucket, target) in enumerate(targets_sorted):
            if i < remainder:
                bucket_targets[bucket] = int(target) + 1
            else:
                bucket_targets[bucket] = int(target)

        low_num = bucket_targets[ActivityBucket.LOW]
        mid_num = bucket_targets[ActivityBucket.MID]
        high_num = bucket_targets[ActivityBucket.HIGH]

        # 候補数をnum*3に拡張して、各バケットから十分な候補を確保
        candidates_per_bucket = num * 3

        # 各バケットの候補を取得（スコア順）
        low_bucketed = [b for b in bucketed_images if b.bucket == ActivityBucket.LOW][
            :candidates_per_bucket
        ]
        mid_bucketed = [b for b in bucketed_images if b.bucket == ActivityBucket.MID][
            :candidates_per_bucket
        ]
        high_bucketed = [b for b in bucketed_images if b.bucket == ActivityBucket.HIGH][
            :candidates_per_bucket
        ]

        # 各バケット内で類似度フィルタリングを適用して選択
        selected: list[ImageMetrics] = []
        total_rejected = 0

        # 全候補数がnum*3より少ない場合は類似度フィルタリングを適用しない
        apply_similarity_filter = len(all_results) >= num * 3

        if apply_similarity_filter:
            # LOWバケットから選択
            low_selected, low_rejected = self._select_with_similarity_filter(
                [b.image for b in low_bucketed], low_num, similarity_threshold
            )
            selected.extend(low_selected)
            total_rejected += low_rejected

            # MIDバケットから選択
            mid_selected, mid_rejected = self._select_with_similarity_filter(
                [b.image for b in mid_bucketed], mid_num, similarity_threshold
            )
            selected.extend(mid_selected)
            total_rejected += mid_rejected

            # HIGHバケットから選択
            high_selected, high_rejected = self._select_with_similarity_filter(
                [b.image for b in high_bucketed], high_num, similarity_threshold
            )
            selected.extend(high_selected)
            total_rejected += high_rejected
        else:
            # 類似度フィルタリングなしで各バケットから選択
            selected.extend([b.image for b in low_bucketed[:low_num]])
            selected.extend([b.image for b in mid_bucketed[:mid_num]])
            selected.extend([b.image for b in high_bucketed[:high_num]])

        # まだ不足する場合は他のバケットから補填
        # バケットバランスを考慮して、未選択のバケットを優先
        if len(selected) < num:
            # 選択されていない画像をバケットごとに分類
            remaining_by_bucket: dict[ActivityBucket, list[ImageMetrics]] = {
                ActivityBucket.LOW: [],
                ActivityBucket.MID: [],
                ActivityBucket.HIGH: [],
            }
            for b in bucketed_images:
                if b.image not in selected:
                    remaining_by_bucket[b.bucket].append(b.image)

            # 各バケットから交互に補填してバランスを維持
            bucket_order = [ActivityBucket.LOW, ActivityBucket.MID, ActivityBucket.HIGH]
            bucket_idx = 0
            while len(selected) < num:
                added = False
                for _ in range(len(bucket_order)):
                    bucket = bucket_order[bucket_idx % len(bucket_order)]
                    bucket_idx += 1
                    if remaining_by_bucket[bucket]:
                        selected.append(remaining_by_bucket[bucket].pop(0))
                        added = True
                        if len(selected) >= num:
                            break
                if not added:
                    break

        # スコア順でソートして返す
        selected.sort(key=lambda x: x.total_score, reverse=True)
        return selected[:num], total_rejected

    def _select_with_similarity_filter(
        self,
        candidates: list[ImageMetrics],
        num: int,
        similarity_threshold: float,
    ) -> tuple[list[ImageMetrics], int]:
        """類似度フィルタリングを適用して画像を選択する.

        Args:
            candidates: 候補画像リスト（スコア降順）
            num: 選択する数
            similarity_threshold: 類似度の閾値

        Returns:
            選択された画像リストと除外された数のタプル
        """
        if not candidates or num <= 0:
            return [], 0

        # 特徴ベクトルを事前にL2正規化
        eps = 1e-8
        normalized_features = []
        for c in candidates:
            norm = np.linalg.norm(c.features)
            if norm < eps:
                normalized_features.append(np.zeros_like(c.features))
            else:
                normalized_features.append(c.features / norm)

        # 段階的しきい値緩和
        threshold_steps = self.config.compute_threshold_steps(similarity_threshold)

        feature_dim = len(normalized_features[0]) if normalized_features else 0
        selected_features_matrix = np.zeros((num, feature_dim), dtype=np.float32)
        selected: list[ImageMetrics] = []
        selected_indices: set[int] = set()

        for threshold in threshold_steps:
            for idx, candidate in enumerate(candidates):
                if idx in selected_indices:
                    continue

                if len(selected) >= num:
                    break

                candidate_feat = normalized_features[idx]

                # 類似度チェック
                is_similar = False
                if selected_indices:
                    selected_count = len(selected)
                    sims = selected_features_matrix[:selected_count] @ candidate_feat
                    if np.any(sims > threshold):
                        is_similar = True
                        # 類似している場合はスキップ
                        continue

                if not is_similar:
                    selected.append(candidate)
                    selected_indices.add(idx)
                    if len(selected) <= num:
                        selected_features_matrix[len(selected) - 1] = candidate_feat

            if len(selected) >= num:
                break

        # 除外数を計算（選択されなかった候補のうち、処理されたもの）
        rejected_count = 0
        for idx, _ in enumerate(candidates):
            if idx not in selected_indices and len(selected) >= num:
                rejected_count += 1

        return selected, rejected_count

    def _bucket_by_activity(self, images: list[ImageMetrics]) -> list[BucketedImage]:
        """画像を活動量に基づいてバケット分けする.

        Args:
            images: 画像メトリクスのリスト

        Returns:
            バケット付けされた画像のリスト
        """
        bucketed: list[BucketedImage] = []

        for img in images:
            activity_score = self._calculate_activity_score(img)
            bucket = self._determine_bucket(activity_score)
            bucketed.append(
                BucketedImage(image=img, bucket=bucket, activity_score=activity_score)
            )

        return bucketed

    def _calculate_activity_score(self, img: ImageMetrics) -> float:
        """活動量スコアを計算する.

        Args:
            img: 画像メトリクス

        Returns:
            活動量スコア（0.0-1.0）
        """
        norm = img.normalized_metrics
        return (
            self.activity_weights.get("action_intensity", 0.55) * norm.action_intensity
            + self.activity_weights.get("edge_density", 0.25) * norm.edge_density
            + self.activity_weights.get("dramatic_score", 0.20) * norm.dramatic_score
        )

    def _determine_bucket(self, activity_score: float) -> ActivityBucket:
        """活動量スコアに基づいてバケットを決定する.

        Args:
            activity_score: 活動量スコア（0.0-1.0）

        Returns:
            活動量バケット
        """
        # テストデータに基づいて境界値を調整
        # LOW: 0.1-0.15 (action_intensity) -> スコア約0.12-0.18
        # MID: 0.5 -> スコア約0.5
        # HIGH: 0.8-0.9 -> スコア約0.8-0.9
        if activity_score < 0.3:
            return ActivityBucket.LOW
        elif activity_score < 0.7:
            return ActivityBucket.MID
        else:
            return ActivityBucket.HIGH
