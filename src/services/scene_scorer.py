"""入力集合内の類似度密度から play / event を割り当てる."""

import numpy as np

from ..constants.scene_label import SceneLabel
from ..models.analyzed_image import AnalyzedImage
from ..models.scene_assessment import SceneAssessment
from ..models.scene_mix import SceneMix
from ..utils.vector_utils import VectorUtils


class SceneScorer:
    """画像集合を play / event に分類する."""

    TOP_K_NEIGHBORS = 5

    def assess_batch(
        self,
        analyzed_images: list[AnalyzedImage],
        scene_mix: SceneMix,
    ) -> list[SceneAssessment]:
        """画像集合をまとめて評価する."""
        if not analyzed_images:
            return []

        density_scores = self._calculate_density_scores(analyzed_images)
        allocation = scene_mix.calculate_allocation(len(analyzed_images))
        play_target = allocation[SceneLabel.PLAY]
        ordered_indices = sorted(
            range(len(analyzed_images)),
            key=lambda index: density_scores[index],
            reverse=True,
        )
        play_indices = set(ordered_indices[:play_target])

        assessments: list[SceneAssessment] = []
        for index, density_score in enumerate(density_scores):
            play_score = density_score
            event_score = 1.0 - density_score
            scene_label = SceneLabel.PLAY if index in play_indices else SceneLabel.EVENT
            assessments.append(
                SceneAssessment(
                    play_score=play_score,
                    event_score=event_score,
                    density_score=density_score,
                    scene_label=scene_label,
                    scene_confidence=abs(play_score - event_score),
                )
            )
        return assessments

    @classmethod
    def _calculate_density_scores(
        cls,
        analyzed_images: list[AnalyzedImage],
    ) -> list[float]:
        """各画像の近傍密度を 0..1 に正規化して返す."""
        if len(analyzed_images) == 1:
            return [1.0]

        normalized_features = np.asarray(
            VectorUtils.normalize_feature_vectors(
                [candidate.combined_features for candidate in analyzed_images]
            ),
            dtype=np.float32,
        )
        neighbor_count = min(cls.TOP_K_NEIGHBORS, len(analyzed_images) - 1)
        raw_scores = np.zeros(len(analyzed_images), dtype=np.float32)

        # 一度の行列乗算で全類似度を計算し、O(n²)を最適化
        similarities_matrix = normalized_features @ normalized_features.T
        np.fill_diagonal(similarities_matrix, -np.inf)
        for index in range(len(analyzed_images)):
            similarities = similarities_matrix[index]
            nearest = np.partition(similarities, -neighbor_count)[-neighbor_count:]
            raw_scores[index] = float(np.mean(nearest))

        min_score = float(raw_scores.min())
        max_score = float(raw_scores.max())
        if np.isclose(min_score, max_score):
            return [0.5 for _ in analyzed_images]
        normalized_scores = (raw_scores - min_score) / (max_score - min_score)
        return [float(score) for score in normalized_scores]
