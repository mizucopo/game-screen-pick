"""Selection Rejection Reason domain contractのtest。"""

from src.video_selection.models.selection_rejection_reason import (
    SelectionRejectionReason,
)


def test_selection_rejection_reasons_have_stable_serialized_values() -> None:
    """未採用の排他的主理由がstableな文字列として列挙されること。

    Arrange:
        - 公開するrejection reasonの文字列一覧が用意される
    Act:
        - Selection Rejection Reasonの値が列挙される
    Assert:
        - schema契約どおりのstable文字列が返されること
    """
    # Arrange
    expected = (
        "title_limit",
        "visual_near_duplicate",
        "similarity_ceiling",
        "spoiler_monotonicity_guard",
        "lower_marginal_utility",
    )

    # Act
    actual = tuple(reason.value for reason in SelectionRejectionReason)

    # Assert
    assert actual == expected
