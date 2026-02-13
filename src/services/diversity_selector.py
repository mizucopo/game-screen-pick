"""多様性を考慮した画像選択ロジック."""

from ..models.image_metrics import ImageMetrics
from ..models.selection_config import SelectionConfig
from ..utils.vector_utils import VectorUtils


class DiversitySelector:
    """多様性を考慮した画像選択クラス.

    類似度に基づいて多様な画像を選択する。
    類似度フィルタリング後、不足する場合はスコア順に補填する。
    """

    def __init__(
        self,
        config: SelectionConfig | None = None,
    ):
        """セレクターを初期化する.

        Args:
            config: 選択設定（Noneの場合はデフォルト値を使用）
        """
        self.config = config or SelectionConfig()

    def select(
        self,
        all_results: list[ImageMetrics],
        num: int,
        similarity_threshold: float,
    ) -> tuple[list[ImageMetrics], int]:
        """多様性を考慮して画像を選択する.

        Args:
            all_results: 解析済みの画像メトリクスリスト（スコア降順ソート済み）
            num: 選択する画像数
            similarity_threshold: 類似度の閾値（これ以上は類似とみなす）

        Returns:
            選択された画像メトリクスのリスト（最大num件、
            有効画像数以下なら必ずnum件）と類似度で除外された数のタプル
        """
        if not all_results:
            return [], 0

        # 全候補を対象にする（固定上位M件の制限を廃止）
        candidates = all_results

        # 特徴ベクトルを事前にL2正規化（コサイン類似度 = 内積になる）
        normalized_features = VectorUtils.normalize_feature_vectors(
            [c.features for c in candidates]
        )

        # 段階的しきい値緩和のステップ
        threshold_steps = self.config.compute_threshold_steps(similarity_threshold)

        # 類似度フィルタリングを実行
        selected_indices, rejected_by_similarity = VectorUtils.select_diverse_indices(
            normalized_features=normalized_features,
            num=num,
            threshold_steps=threshold_steps,
        )

        # 最終フォールバック：まだ不足する場合は未選択候補を総合スコア順で埋める
        # （類似度制約を外すため、rejected_indicesも考慮対象に含める）
        if len(selected_indices) < num:
            for idx, _candidate in enumerate(candidates):
                if idx not in selected_indices:
                    selected_indices.add(idx)
                    if len(selected_indices) >= num:
                        break

        # スコア順でソートして返す
        selected = [candidates[i] for i in selected_indices]
        selected.sort(key=lambda x: x.total_score, reverse=True)
        return selected[:num], rejected_by_similarity
