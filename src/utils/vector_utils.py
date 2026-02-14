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
    ) -> tuple[set[int], int]:
        """類似度に基づいて多様なインデックスを選択する.

        類似度で除外された数も正しく計測する。

        Args:
            normalized_features: L2正規化された特徴ベクトルのリスト
            num: 選択する数
            threshold_steps: 段階的なしきい値のリスト（厳しい順）

        Returns:
            (選択されたインデックスのセット, 類似度で除外された数) のタプル

        Note:
            除外数は、類似度チェックによって最終的に除外された候補数のみをカウント。
            容量制約で未選択になったものは含まない。
        """
        feature_dim = len(normalized_features[0]) if normalized_features else 0
        selected_features_matrix = np.zeros((num, feature_dim), dtype=np.float32)
        selected_indices: set[int] = set()
        # 各候補が「類似度によって拒否されたか」を追跡
        # （複数ステップで重複カウントしないようsetで管理）
        rejected_by_similarity_set: set[int] = set()
        selected_count = 0

        # 容量制約または類似度チェックで全候補を走査
        for threshold in threshold_steps:
            for idx, candidate_feat in enumerate(normalized_features):
                if idx in selected_indices:
                    continue

                if selected_count >= num:
                    break

                # 類似度チェック
                is_similar = False
                if selected_indices:
                    sims = selected_features_matrix[:selected_count] @ candidate_feat
                    if np.any(sims > threshold):
                        is_similar = True
                        rejected_by_similarity_set.add(idx)

                if not is_similar:
                    selected_features_matrix[selected_count] = candidate_feat
                    selected_indices.add(idx)
                    selected_count += 1

            if selected_count >= num:
                break

        # 類似度で一度拒否され、最終的に選択されなかった候補数をカウント
        # （後続ステップで選択されたものは除外集合から除外）
        rejected_by_similarity = len(rejected_by_similarity_set - selected_indices)

        return selected_indices, rejected_by_similarity

    @staticmethod
    def filter_by_similarity(
        candidates: list[np.ndarray[Any, Any]],
        num: int,
        similarity_threshold: float,
        compute_threshold_steps: Callable[[float], list[float]],
    ) -> tuple[set[int], int]:
        """類似度に基づいて候補をフィルタリングする.

        特徴ベクトルの正規化、しきい値ステップの計算、多様なインデックスの選択を
        一連の処理として実行する。

        Args:
            candidates: 候補の特徴ベクトルリスト
            num: 選択する数
            similarity_threshold: 類似度の閾値
            compute_threshold_steps: しきい値からステップリストを計算する関数

        Returns:
            (選択されたインデックスのセット, 類似度で除外された数) のタプル
        """
        normalized_features = VectorUtils.normalize_feature_vectors(candidates)
        threshold_steps = compute_threshold_steps(similarity_threshold)
        return VectorUtils.select_diverse_indices(
            normalized_features=normalized_features,
            num=num,
            threshold_steps=threshold_steps,
        )
