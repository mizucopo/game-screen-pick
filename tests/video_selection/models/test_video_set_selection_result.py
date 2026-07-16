"""Video Set Selection Result domain contractのtest。"""

from src.video_selection.models.rejected_blog_candidate import (
    RejectedBlogCandidate,
)
from src.video_selection.models.selected_blog_image import SelectedBlogImage
from src.video_selection.models.selection_rejection_reason import (
    SelectionRejectionReason,
)
from src.video_selection.models.video_set_selection_result import (
    VideoSetSelectionResult,
)
from tests.video_selection.fakes.selection_model_factory import (
    build_blog_candidate,
    build_selection_score,
)


def test_result_summarizes_shortfall_spoilers_and_rejections() -> None:
    """選定結果からshortfall・Major Spoiler件数・rejection件数が集計されること。

    Arrange:
        - Major Spoilerを含む選択画像2件と未採用候補2件が用意される
        - 要求枚数が3枚に設定される
    Act:
        - Video Set Selection Resultのsummary propertyが読み出される
    Assert:
        - shortfall、Major Spoiler件数、reason別件数が返されること
    """
    # Arrange
    score = build_selection_score()
    selected = tuple(
        SelectedBlogImage(
            candidate=candidate,
            selection_index=index,
            score=score,
            reason_codes=(),
            variant_group_id=f"variant_{index}",
            tie_break_applied=False,
        )
        for index, candidate in enumerate(
            (
                build_blog_candidate("a"),
                build_blog_candidate("b", spoiler_risk="high"),
            ),
            start=1,
        )
    )
    rejected = tuple(
        RejectedBlogCandidate(
            candidate=build_blog_candidate(digest),
            reason_code=reason,
            counterfactual_score=score,
            blocked_by_image_id=None,
            nearest_selected_image_id=None,
            similarity=None,
            variant_group_id=f"variant_{digest}",
        )
        for digest, reason in (
            ("c", SelectionRejectionReason.SIMILARITY_CEILING),
            ("d", SelectionRejectionReason.LOWER_MARGINAL_UTILITY),
        )
    )
    result = VideoSetSelectionResult(
        selected=selected,
        rejected=rejected,
        requested_count=3,
        blog_image_type_targets={"normal_gameplay": 2, "event": 1},
        blog_image_type_actuals={"normal_gameplay": 2, "event": 0},
        final_similarity_ceiling=0.98,
        major_spoiler_limit=1,
        annotated_candidate_count=4,
        shortlist_expansion_count=0,
        all_candidate_moments_exhausted=True,
    )

    # Act
    shortfall = result.shortfall
    major_spoiler_count = result.major_spoiler_selected_count
    rejection_counts = result.rejection_counts

    # Assert
    assert shortfall is True
    assert major_spoiler_count == 1
    assert rejection_counts == {
        "similarity_ceiling": 1,
        "lower_marginal_utility": 1,
    }
