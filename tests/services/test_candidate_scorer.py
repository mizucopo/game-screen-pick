"""CandidateScorer の単体テスト."""

from src.analyzers.metric_calculator import MetricCalculator
from src.constants.selection_profiles import ACTIVE_PROFILE
from src.models.analyzer_config import AnalyzerConfig
from src.models.scene_assessment import SceneAssessment
from src.services.candidate_scorer import CandidateScorer
from tests.conftest import create_analyzed_image


def test_candidate_scorer_combines_quality_and_scene_confidence() -> None:
    """品質スコアとscene confidenceからselection_scoreが計算されること.

    Arrange:
        - CandidateScorerと分析済み画像がある
        - Ollama分類済みのSceneAssessmentがある
    Act:
        - 候補をスコアリングする
    Assert:
        - selection_scoreが品質と分類信頼度の合成値になること
    """
    # Arrange
    scorer = CandidateScorer(MetricCalculator(AnalyzerConfig()))
    image = create_analyzed_image(path="/tmp/frame.jpg")

    # Act
    candidate = scorer.score(
        analyzed_image=image,
        assessment=SceneAssessment(
            scene_slug="battle",
            scene_display_name="戦闘",
            scene_description="敵との戦闘場面",
            scene_confidence=0.60,
        ),
        profile=ACTIVE_PROFILE,
    )

    # Assert
    expected_score = (candidate.quality_score * 0.7) + (0.60 * 0.3)
    assert candidate.selection_score == expected_score
    assert 0.0 <= candidate.quality_score <= 1.0
