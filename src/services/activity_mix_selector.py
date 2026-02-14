"""活動量ミックスを考慮した画像選択ロジック."""

from ..models.activity_bucket import ActivityBucket
from ..models.bucketed_image import BucketedImage
from ..models.image_metrics import ImageMetrics
from ..models.selection_config import SelectionConfig
from ..utils.vector_utils import VectorUtils


class ActivityMixSelector:
    """活動量ミックスを考慮した画像選択クラス.

    活動量（LOW/MID/HIGH）に基づいてバランスよく画像を選択し、
    類似度フィルタリングも適用する。バケットごとのターゲット数を
    比率配分で計算し、不足する場合は他バケットから補填する。
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

        全体多様性を確保するため、類似度制約を全体集合に先に適用し、
        その後に活動量バケットごとの配分を行う。

        Args:
            all_results: 解析済みの画像メトリクスリスト（スコア降順ソート済み）
            num: 選択する画像数
            similarity_threshold: 類似度の閾値

        Returns:
            選択された画像メトリクスのリストと類似度で除外された数のタプル
        """
        if not all_results:
            return [], 0

        # 類似度フィルタの適用条件:
        # 候補数がnum*3以上の場合のみ適用
        # 理由: 十分な候補がない場合に類似度で除外すると、選択数を満たせなくなるため
        apply_similarity_filter = len(all_results) >= num * 3

        if apply_similarity_filter:
            # ステップ1: 類似度フィルタを全体集合に適用（バケット間重複を抑制）
            # 類似度制約を満たす候補プールを構築
            filtered_pool, rejected_count = self._filter_by_similarity_overall(
                all_results, num * 3, similarity_threshold
            )
            # フィルタ済みプールが不足する場合は、all_resultsから補完用プールを追加
            pool_for_bucketing = list(filtered_pool)
            already_in_pool = {id(img) for img in filtered_pool}
            if len(pool_for_bucketing) < num:
                for img in all_results:
                    if id(img) not in already_in_pool:
                        pool_for_bucketing.append(img)
                        already_in_pool.add(id(img))
                        if len(pool_for_bucketing) >= num * 3:
                            break
        else:
            # 類似度フィルタなし
            pool_for_bucketing = all_results
            rejected_count = 0

        # ステップ2: プールを活動量スコアでバケット分け
        bucketed_images = self._bucket_by_activity(pool_for_bucketing)

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

        # 各バケットの候補を取得してスコア降順にソート（品質優先）
        low_bucketed = sorted(
            (b for b in bucketed_images if b.bucket == ActivityBucket.LOW),
            key=lambda b: b.image.total_score,
            reverse=True,
        )
        mid_bucketed = sorted(
            (b for b in bucketed_images if b.bucket == ActivityBucket.MID),
            key=lambda b: b.image.total_score,
            reverse=True,
        )
        high_bucketed = sorted(
            (b for b in bucketed_images if b.bucket == ActivityBucket.HIGH),
            key=lambda b: b.image.total_score,
            reverse=True,
        )

        # 各バケットからターゲット数を選択
        selected: list[ImageMetrics] = []
        selected.extend([b.image for b in low_bucketed[:low_num]])
        selected.extend([b.image for b in mid_bucketed[:mid_num]])
        selected.extend([b.image for b in high_bucketed[:high_num]])

        # まだ不足する場合は他のバケットから補填
        if len(selected) < num:
            # 選択されていない画像をバケットごとに分類
            remaining_by_bucket: dict[ActivityBucket, list[ImageMetrics]] = {
                ActivityBucket.LOW: [],
                ActivityBucket.MID: [],
                ActivityBucket.HIGH: [],
            }
            # id ベースの集合で「選択済み」を判定
            # 理由: ImageMetricsの__eq__はvalueベースの比較を実装しており、
            #       集合演算中にValueErrorを引き起こす可能性があるため、
            #       identity(id)ベースで判定することで安全に除外チェックを行う
            selected_ids = {id(img) for img in selected}
            # バケットごとにスコア順に収集
            for bucket in ActivityBucket:
                candidates = [
                    b.image
                    for b in bucketed_images
                    if b.bucket == bucket and id(b.image) not in selected_ids
                ]
                remaining_by_bucket[bucket] = sorted(
                    candidates,
                    key=lambda img: img.total_score,
                    reverse=True,
                )

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
        return selected[:num], rejected_count

    def _filter_by_similarity_overall(
        self,
        candidates: list[ImageMetrics],
        max_pool_size: int,
        similarity_threshold: float,
    ) -> tuple[list[ImageMetrics], int]:
        """全体集合で類似度フィルタリングを適用し、多様な候補プールを構築する.

        Args:
            candidates: 候補画像リスト（スコア降順）
            max_pool_size: プールの最大サイズ
            similarity_threshold: 類似度の閾値

        Returns:
            フィルタ済みの画像リストと除外された数のタプル
        """
        if not candidates or max_pool_size <= 0:
            return [], 0

        # 特徴ベクトルを事前にL2正規化
        normalized_features = VectorUtils.normalize_feature_vectors(
            [c.features for c in candidates]
        )

        # 段階的しきい値緩和
        threshold_steps = self.config.compute_threshold_steps(similarity_threshold)

        # 類似度フィルタリングを実行
        selected_indices, rejected_by_similarity = VectorUtils.select_diverse_indices(
            normalized_features=normalized_features,
            num=max_pool_size,
            threshold_steps=threshold_steps,
        )

        selected = [candidates[i] for i in sorted(selected_indices)]
        return selected, rejected_by_similarity

    def _bucket_by_activity(self, images: list[ImageMetrics]) -> list[BucketedImage]:
        """画像を活動量に基づいてバケット分けする.

        quantileモードの場合、順位（パーセンタイル）に基づいて3等分し、
        同値スコアによる偏りを防ぐ。画像数に応じてバケットサイズが調整される。

        Args:
            images: 画像メトリクスのリスト

        Returns:
            バケット付けされた画像のリスト
        """
        # すべての活動量スコアを計算し、スコア順にソート
        score_image_pairs: list[tuple[float, ImageMetrics]] = []
        for img in images:
            activity_score = self._calculate_activity_score(img)
            score_image_pairs.append((activity_score, img))

        # 順位に基づいてバケット分け（同値スコアによる偏りを防止）
        bucketed: list[BucketedImage] = []
        if score_image_pairs:
            sorted_pairs = sorted(score_image_pairs, key=lambda x: x[0])
            n = len(sorted_pairs)
            # 順位に基づいて3等分
            low_end = n // 3
            high_end = 2 * n // 3

            for rank, (activity_score, img) in enumerate(sorted_pairs):
                if rank < low_end:
                    bucket = ActivityBucket.LOW
                elif rank < high_end:
                    bucket = ActivityBucket.MID
                else:
                    bucket = ActivityBucket.HIGH
                bucketed.append(
                    BucketedImage(
                        image=img, bucket=bucket, activity_score=activity_score
                    )
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
