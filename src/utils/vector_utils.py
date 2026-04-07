"""ベクトル操作ユーティリティ."""

from collections.abc import Callable
from typing import Any

import numpy as np


class VectorUtils:
    """ベクトル操作に関するユーティリティクラス.

    画像特徴ベクトルの正規化と類似度フィルタリング機能を提供する。
    """

    @staticmethod
    def safe_l2_normalize(
        vec: np.ndarray[Any, Any], eps: float = 1e-8
    ) -> np.ndarray[Any, Any]:
        """ゼロ割れ安全なL2正規化を行う.

        Args:
            vec: 正規化するベクトル
            eps: ゼロ割れ防止用の微小値

        Returns:
            L2正規化されたベクトル（元のノルムが0の場合はゼロベクトル）
        """
        norm = float(np.linalg.norm(vec))
        if norm < eps:
            return np.zeros_like(vec)
        return vec / norm

    @staticmethod
    def normalize_feature_vectors(
        vectors: list[np.ndarray[Any, Any]], eps: float = 1e-8
    ) -> list[np.ndarray[Any, Any]]:
        """特徴ベクトルリストをL2正規化する.

        Args:
            vectors: 正規化する特徴ベクトルのリスト
            eps: ゼロ割れ防止用の微小値

        Returns:
            L2正規化された特徴ベクトルのリスト
        """
        normalized = []
        for vec in vectors:
            norm = np.linalg.norm(vec)
            if norm < eps:
                normalized.append(np.zeros_like(vec))
            else:
                normalized.append(vec / norm)
        return normalized

    @staticmethod
    def select_diverse_indices(
        normalized_features: list[np.ndarray[Any, Any]],
        num: int,
        threshold_steps: list[float],
        seed_features: list[np.ndarray[Any, Any]] | None = None,
    ) -> tuple[list[int], set[int]]:
        """類似度に基づいて多様なインデックスを選択する.

        既に選択済みの特徴ベクトルをseedとして受け取り、
        新規候補との類似度をまとめて判定できる。

        Args:
            normalized_features: L2正規化された特徴ベクトルのリスト
            num: 選択する数
            threshold_steps: 段階的なしきい値のリスト（厳しい順）
            seed_features: 既に採用済みの正規化特徴ベクトル。
                これら自体は返り値に含めず、類似度比較の基準としてのみ使う。

        Returns:
            (選択されたインデックスのリスト, 類似度で除外された
            インデックス集合) のタプル

        Note:
            除外集合には、類似度チェックで一度拒否され、
            最終的にも選択されなかった候補だけが残る。
            容量制約で未選択になったものは含まない。
        """
        if num <= 0 or not normalized_features:
            return [], set()

        seed_features = seed_features or []
        feature_dim = len(normalized_features[0])
        target_count = min(num, len(normalized_features))
        selected_features_matrix = np.zeros(
            (len(seed_features) + target_count, feature_dim),
            dtype=np.float32,
        )
        selected_indices: list[int] = []
        selected_index_set: set[int] = set()
        rejected_by_similarity_set: set[int] = set()

        for idx, seed_feature in enumerate(seed_features):
            selected_features_matrix[idx] = seed_feature
        selected_count = len(seed_features)

        # 容量制約または類似度チェックで全候補を走査
        for threshold in threshold_steps:
            for idx, candidate_feat in enumerate(normalized_features):
                if idx in selected_index_set:
                    continue
                if idx in rejected_by_similarity_set:
                    continue

                if len(selected_indices) >= target_count:
                    break

                # 類似度チェック
                is_similar = False
                if selected_count > 0:
                    sims = selected_features_matrix[:selected_count] @ candidate_feat
                    if np.any(sims > threshold):
                        is_similar = True
                        rejected_by_similarity_set.add(idx)

                if not is_similar:
                    selected_features_matrix[selected_count] = candidate_feat
                    selected_indices.append(idx)
                    selected_index_set.add(idx)
                    selected_count += 1

            if len(selected_indices) >= target_count:
                break

        return selected_indices, rejected_by_similarity_set - selected_index_set

    @staticmethod
    def filter_by_similarity(
        candidates: list[np.ndarray[Any, Any]],
        num: int,
        similarity_threshold: float,
        compute_threshold_steps: Callable[[float], list[float]],
        seed_features: list[np.ndarray[Any, Any]] | None = None,
    ) -> tuple[list[int], set[int]]:
        """類似度に基づいて候補をフィルタリングする.

        特徴ベクトルの正規化、しきい値ステップの計算、多様なインデックスの選択を
        一連の処理として実行する。

        Args:
            candidates: 候補の特徴ベクトルリスト
            num: 選択する数
            similarity_threshold: 類似度の閾値
            compute_threshold_steps: しきい値からステップリストを計算する関数
            seed_features: 既に選択済み候補の特徴ベクトルリスト

        Returns:
            (選択されたインデックスのリスト, 類似度で除外された
            インデックス集合) のタプル
        """
        normalized_features = VectorUtils.normalize_feature_vectors(candidates)
        normalized_seed_features = VectorUtils.normalize_feature_vectors(
            seed_features or []
        )
        threshold_steps = compute_threshold_steps(similarity_threshold)
        return VectorUtils.select_diverse_indices(
            normalized_features=normalized_features,
            num=num,
            threshold_steps=threshold_steps,
            seed_features=normalized_seed_features,
        )
