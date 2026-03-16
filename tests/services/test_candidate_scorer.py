"""CandidateScorer の単体テスト."""

from src.analyzers.metric_calculator import MetricCalculator
from src.constants.scene_label import SceneLabel
from src.constants.selection_profiles import ACTIVE_PROFILE
from src.models.analyzer_config import AnalyzerConfig
from src.models.scene_assessment import SceneAssessment
from src.services.candidate_scorer import CandidateScorer
from tests.conftest import create_analyzed_image


def test_candidate_scorer_penalizes_transition_like_low_confidence_candidate() -> None:
    """veiled transition は selection_score を明確に下げること."""
    scorer = CandidateScorer(MetricCalculator(AnalyzerConfig()))
    image = create_analyzed_image(path="/tmp/event0039.jpg")
    scene_label = SceneLabel.EVENT

    clean = scorer.score(
        analyzed_image=image,
        assessment=SceneAssessment(
            gameplay_score=0.20,
            event_score=0.52,
            other_score=0.18,
            scene_label=scene_label,
            scene_confidence=0.07,
            transition_risk_score=0.10,
            bright_washout_score=0.05,
            veiled_transition_score=0.05,
            transition_suppressed_event=False,
        ),
        profile=ACTIVE_PROFILE,
        information_score=0.70,
        distinctiveness_score=0.72,
    )
    bad = scorer.score(
        analyzed_image=image,
        assessment=SceneAssessment(
            gameplay_score=0.26,
            event_score=0.40,
            other_score=0.36,
            scene_label=scene_label,
            scene_confidence=0.01,
            transition_risk_score=0.56,
            bright_washout_score=0.58,
            veiled_transition_score=0.62,
            transition_suppressed_event=True,
        ),
        profile=ACTIVE_PROFILE,
        information_score=0.42,
        distinctiveness_score=0.62,
    )

    assert bad.selection_score < clean.selection_score
    assert clean.selection_score - bad.selection_score > 10.0
