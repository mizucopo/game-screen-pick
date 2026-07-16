"""Rejected Blog Candidate domain contractのtest。"""

from src.video_selection.models.rejected_blog_candidate import (
    RejectedBlogCandidate,
)
from src.video_selection.models.selection_rejection_reason import (
    SelectionRejectionReason,
)
from tests.video_selection.fakes.selection_model_factory import (
    build_blog_candidate,
    build_selection_score,
)


def test_rejected_blog_candidate_keeps_counterfactual_and_blocking_evidence() -> None:
    """未採用候補の反実仮想scoreとblocking evidenceが保持されること。

    Arrange:
        - Blog Candidate、反実仮想score、similarity blockerが用意される
    Act:
        - Rejected Blog Candidateが構築される
    Assert:
        - stable reason、blocking ID、similarity、Variant Groupが返されること
    """
    # Arrange
    candidate = build_blog_candidate()
    score = build_selection_score()

    # Act
    rejected = RejectedBlogCandidate(
        candidate=candidate,
        reason_code=SelectionRejectionReason.SIMILARITY_CEILING,
        counterfactual_score=score,
        blocked_by_image_id=None,
        nearest_selected_image_id="frm_" + "b" * 64,
        similarity=0.9,
        variant_group_id="variant_test",
    )

    # Assert
    assert rejected.candidate.identifier == candidate.identifier
    assert rejected.reason_code is SelectionRejectionReason.SIMILARITY_CEILING
    assert rejected.counterfactual_score == score
    assert rejected.blocked_by_image_id is None
    assert rejected.nearest_selected_image_id == "frm_" + "b" * 64
    assert rejected.similarity == 0.9
    assert rejected.variant_group_id == "variant_test"
