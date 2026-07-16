"""Selection Score domain contractのtest。"""

from src.video_selection.models.selection_score import SelectionScore


def test_selection_score_exposes_reproducible_numeric_components() -> None:
    """選定判断の全数値componentが再現可能な形で保持されること。

    Arrange:
        - utility、penalty、similarityの数値内訳が用意される
    Act:
        - Selection Scoreが構築され数値componentが読み出される
    Assert:
        - 入力した全数値componentが同じ値で返されること
    """
    # Arrange
    expected = (0.81, 0.1, 0.05, 0.02, 0.74, 0.78, 0.75)

    # Act
    score = SelectionScore(
        base_utility=expected[0],
        spoiler_penalty=expected[1],
        coverage_bonus=expected[2],
        temporal_diversity_penalty=expected[3],
        marginal_utility=expected[4],
        similarity_pass=expected[5],
        nearest_selected_similarity=expected[6],
    )
    actual = (
        score.base_utility,
        score.spoiler_penalty,
        score.coverage_bonus,
        score.temporal_diversity_penalty,
        score.marginal_utility,
        score.similarity_pass,
        score.nearest_selected_similarity,
    )

    # Assert
    assert actual == expected
