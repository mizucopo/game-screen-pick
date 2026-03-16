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
        play_target = self._calculate_play_target(len(analyzed_images), scene_mix)
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
            scene_label = (
                SceneLabel.PLAY if index in play_indices else SceneLabel.EVENT
            )
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

        for index, feature in enumerate(normalized_features):
            similarities = normalized_features @ feature
            similarities[index] = -np.inf
            nearest = np.partition(similarities, -neighbor_count)[-neighbor_count:]
            raw_scores[index] = float(np.mean(nearest))

        min_score = float(raw_scores.min())
        max_score = float(raw_scores.max())
        if np.isclose(min_score, max_score):
            return [0.5 for _ in analyzed_images]
        normalized_scores = (raw_scores - min_score) / (max_score - min_score)
        return [float(score) for score in normalized_scores]

    @staticmethod
    def _calculate_play_target(total: int, scene_mix: SceneMix) -> int:
        """play 枚数の目標値を返す."""
        raw_play = total * scene_mix.play
        raw_event = total * scene_mix.event
        base_play = int(raw_play)
        base_event = int(raw_event)
        remainder = total - (base_play + base_event)
        if remainder <= 0:
            return base_play
        return base_play + int(raw_play - base_play >= raw_event - base_event)
