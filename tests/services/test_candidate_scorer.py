"""CandidateScorer の単体テスト."""

from src.analyzers.metric_calculator import MetricCalculator
from src.constants.scene_label import SceneLabel
from src.constants.selection_profiles import ACTIVE_PROFILE
from src.models.analyzer_config import AnalyzerConfig
from src.models.scene_assessment import SceneAssessment
from src.services.candidate_scorer import CandidateScorer
from tests.conftest import create_analyzed_image


def test_candidate_scorer_uses_scene_specific_score() -> None:
    """play/event で selection_score の参照先が切り替わること.

    Given:
        - CandidateScorerと分析済み画像がある
        - play/eventそれぞれのSceneAssessmentがある
    When:
        - playとeventの候補をスコアリングする
    Then:
        - play候補はplay_scoreがselection_scoreになること
        - event候補はevent_scoreがselection_scoreになること
    """
    # Arrange
    scorer = CandidateScorer(MetricCalculator(AnalyzerConfig()))
    image = create_analyzed_image(path="/tmp/frame.jpg")

    # Act
    play_candidate = scorer.score(
        analyzed_image=image,
        assessment=SceneAssessment(
            play_score=0.72,
            event_score=0.28,
            density_score=0.72,
            scene_label=SceneLabel.PLAY,
            scene_confidence=0.44,
        ),
        profile=ACTIVE_PROFILE,
    )
    event_candidate = scorer.score(
        analyzed_image=image,
        assessment=SceneAssessment(
            play_score=0.20,
            event_score=0.80,
            density_score=0.20,
            scene_label=SceneLabel.EVENT,
            scene_confidence=0.60,
        ),
        profile=ACTIVE_PROFILE,
    )

    # Assert
    assert play_candidate.selection_score == 0.72
    assert event_candidate.selection_score == 0.80
    assert 0.0 <= play_candidate.quality_score <= 1.0
