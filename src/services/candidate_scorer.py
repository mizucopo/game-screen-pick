"""中立解析結果を最終候補へ変換する採点器."""

from ..analyzers.metric_calculator import MetricCalculator
from ..constants.scene_label import SceneLabel
from ..models.analyzed_image import AnalyzedImage
from ..models.scene_assessment import SceneAssessment
from ..models.scored_candidate import ScoredCandidate
from ..models.selection_profile import SelectionProfile


class CandidateScorer:
    """プロファイル別の品質・選定スコアを計算する."""

    def __init__(self, metric_calculator: MetricCalculator):
        """CandidateScorerを初期化する."""
        self.metric_calculator = metric_calculator

    def score(
        self,
        analyzed_image: AnalyzedImage,
        assessment: SceneAssessment,
        profile: SelectionProfile,
    ) -> ScoredCandidate:
        """中立解析結果を最終候補に変換する."""
        quality_score = self.metric_calculator.calculate_quality_score(
            analyzed_image.normalized_metrics,
            profile.quality_weights,
        )
        selection_score = (
            assessment.play_score
            if assessment.scene_label == SceneLabel.PLAY
            else assessment.event_score
        )
        return ScoredCandidate(
            analyzed_image=analyzed_image,
            scene_assessment=assessment,
            resolved_profile=profile.name,
            quality_score=quality_score,
            selection_score=selection_score,
        )
