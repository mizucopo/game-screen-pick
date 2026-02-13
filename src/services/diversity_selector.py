"""多様性を考慮した画像選択ロジック."""

import random

import numpy as np

from ..models.image_metrics import ImageMetrics
from ..models.selection_config import SelectionConfig


class DiversitySelector:
    """多様性を考慮した画像選択クラス.

    類似度に基づいて多様な画像を選択する。
    """

    def __init__(
        self,
        config: SelectionConfig | None = None,
        rng: random.Random | None = None,
    ):
        """セレクターを初期化する.

        Args:
            config: 選択設定（Noneの場合はデフォルト値を使用）
            rng: 乱数生成器（Noneの場合はデフォルトのRandomを使用）
        """
        self.config = config or SelectionConfig()
        self._rng = rng or random.Random()

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
        eps = 1e-8
        normalized_features = []
        for c in candidates:
            norm = np.linalg.norm(c.features)
            if norm < eps:
                # ゼロノルムの場合はゼロベクトルとして扱う
                normalized_features.append(np.zeros_like(c.features))
            else:
                normalized_features.append(c.features / norm)

        # 段階的しきい値緩和のステップ
        threshold_steps = self.config.compute_threshold_steps(similarity_threshold)

        # 選択済み特徴を保持する行列（事前に最大サイズ確保）
        feature_dim = len(normalized_features[0]) if normalized_features else 0
        selected_features_matrix = np.zeros((num, feature_dim), dtype=np.float32)
        selected: list[ImageMetrics] = []
        selected_indices: set[int] = set()
        rejected_indices: set[int] = set()  # ユニークな拒否数を追跡

        # 各しきい値で選択を試行
        for threshold in threshold_steps:
            for idx, candidate in enumerate(candidates):
                # 既に選択または永続拒否された候補はスキップ
                if idx in selected_indices or idx in rejected_indices:
                    continue

                if len(selected) >= num:
                    break

                candidate_feat = normalized_features[idx]

                # 既に選ばれた画像たちと「見た目」を比較（事前確保行列で効率化）
                is_similar = False
                if selected_indices:
                    # selected_count分だけ行列のスライスを使用して類似度を計算
                    selected_count = len(selected)
                    sims = selected_features_matrix[:selected_count] @ candidate_feat
                    if np.any(sims > threshold):
                        is_similar = True
                        # 類似している場合はスキップして次の候補へ
                        continue

                if not is_similar:
                    selected.append(candidate)
                    selected_indices.add(idx)
                    # 事前確保した行列に特徴を追加
                    if len(selected) <= num:
                        selected_features_matrix[len(selected) - 1] = candidate_feat

            if len(selected) >= num:
                break

        # 最終フォールバック：まだ不足する場合は未選択候補を総合スコア順で埋める
        # （類似度制約を外す）
        if len(selected) < num:
            for idx, candidate in enumerate(candidates):
                if idx not in selected_indices and idx not in rejected_indices:
                    selected.append(candidate)
                    selected_indices.add(idx)
                    if len(selected) >= num:
                        break

        # スコア順でソートして返す
        selected.sort(key=lambda x: x.total_score, reverse=True)
        return selected[:num], len(rejected_indices)
