"""中立解析結果を最終候補へ変換する採点器."""

from ..analyzers.metric_calculator import MetricCalculator
from ..constants.scene_label import SceneLabel
from ..models.analyzed_image import AnalyzedImage
from ..models.scene_assessment import SceneAssessment
from ..models.scored_candidate import ScoredCandidate
from ..models.selection_profile import SelectionProfile
from ..utils.transition_metrics import clamp01


class CandidateScorer:
    """プロファイル別の品質・活動量・選定スコアを計算する."""

    def __init__(self, metric_calculator: MetricCalculator):
        """CandidateScorerを初期化する.

        Args:
            metric_calculator: 画質スコアと明度ペナルティを計算する器。
        """
        self.metric_calculator = metric_calculator

    def score(
        self,
        analyzed_image: AnalyzedImage,
        assessment: SceneAssessment,
        profile: SelectionProfile,
        information_score: float,
        distinctiveness_score: float,
    ) -> ScoredCandidate:
        """中立解析結果を最終候補に変換する.

        品質スコアは画質メトリクスのみから算出し、
        活動量スコアは action / edge / UI / gameplay score を
        プロファイル重みで合成する。最終的な `selection_score` は
        scene mix向けスコア、品質スコア、入力全体に対する
        情報量スコア、差分量スコアを固定重みで合成する。

        Args:
            analyzed_image: 中立解析済みの画像データ。
            assessment: scene判定結果。
            profile: 選択時に使う解決済みプロファイル。
            information_score: 入力全体に対する相対情報量スコア。
            distinctiveness_score: 入力全体に対する相対差分量スコア。

        Returns:
            選定に必要な全スコアを持つ `ScoredCandidate` 。
        """
        quality_score = self.metric_calculator.calculate_quality_score(
            analyzed_image.normalized_metrics,
            profile.quality_weights,
        )
        activity_score = (
            profile.activity_weights["action_intensity"]
            * analyzed_image.normalized_metrics.action_intensity
            + profile.activity_weights["edge_density"]
            * analyzed_image.normalized_metrics.edge_density
            + profile.activity_weights["ui_density"]
            * analyzed_image.normalized_metrics.ui_density
            + profile.activity_weights["gameplay_score"] * assessment.gameplay_score
        )
        scene_mix_score = self._calculate_scene_mix_score(assessment)
        base_selection_score = max(
            0.0,
            100.0
            * (
                0.50 * scene_mix_score
                + 0.25 * quality_score
                + 0.15 * information_score
                + 0.10 * distinctiveness_score
            ),
        )
        low_confidence_factor = clamp01((0.08 - assessment.scene_confidence) / 0.08)
        transition_selection_penalty = (
            14.0 * assessment.veiled_transition_score
            + 8.0 * assessment.veiled_transition_score * low_confidence_factor
        )
        if assessment.transition_suppressed_event:
            transition_selection_penalty += 6.0
        relative_low_confidence_factor = clamp01(
            (0.06 - assessment.scene_confidence) / 0.06
        )
        relative_transition_penalty = (
            18.0 * assessment.relative_transition_score
            + 10.0
            * assessment.relative_transition_score
            * relative_low_confidence_factor
        )
        if (
            assessment.relative_transition_polarity == "bright"
            and assessment.relative_bright_transition_score >= 0.62
        ):
            relative_transition_penalty += 6.0
        if (
            assessment.relative_transition_polarity == "dark"
            and assessment.relative_dark_transition_score >= 0.60
        ):
            relative_transition_penalty += 5.0
        selection_score = max(
            0.0,
            base_selection_score
            - transition_selection_penalty
            - relative_transition_penalty,
        )
        return ScoredCandidate(
            analyzed_image=analyzed_image,
            scene_assessment=assessment,
            resolved_profile=profile.name,
            quality_score=quality_score,
            activity_score=activity_score,
            selection_score=selection_score,
        )

    @staticmethod
    def _calculate_scene_mix_score(assessment: SceneAssessment) -> float:
        """画面種別に応じた選定用scene scoreを返す.

        gameplay / event / other のどのbucketに属するかで、
        各scene scoreの重み付けを切り替える。
        これにより、同じ品質でもそのbucketらしい画像が上に来やすくなる。

        Args:
            assessment: scene判定結果。

        Returns:
            selection score の土台になるscene mix向けスコア。
        """
        if assessment.scene_label == SceneLabel.GAMEPLAY:
            return (
                0.75 * assessment.gameplay_score
                + 0.20 * assessment.event_score
                + 0.05 * assessment.other_score
            )
        if assessment.scene_label == SceneLabel.EVENT:
            return (
                0.20 * assessment.gameplay_score
                + 0.75 * assessment.event_score
                + 0.05 * assessment.other_score
            )
        return (
            0.15 * assessment.gameplay_score
            + 0.15 * assessment.event_score
            + 0.70 * assessment.other_score
        )
