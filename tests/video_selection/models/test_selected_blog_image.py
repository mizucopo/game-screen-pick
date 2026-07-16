"""Selected Blog Image domain contractのtest。"""

from src.video_selection.models.selected_blog_image import SelectedBlogImage
from tests.video_selection.fakes.selection_model_factory import (
    build_blog_candidate,
    build_selection_score,
)


def test_selected_blog_image_keeps_order_reasons_and_variant_provenance() -> None:
    """採用画像の順序・理由・Variant provenanceが保持されること。

    Arrange:
        - Blog Candidate、Selection Score、安定した選定metadataが用意される
    Act:
        - Selected Blog Imageが構築される
    Assert:
        - 選択順、理由、Variant Group、tie-break使用有無が返されること
    """
    # Arrange
    candidate = build_blog_candidate()
    score = build_selection_score()

    # Act
    selected = SelectedBlogImage(
        candidate=candidate,
        selection_index=2,
        score=score,
        reason_codes=("high_quality", "stable_tie_break"),
        variant_group_id="variant_test",
        tie_break_applied=True,
    )

    # Assert
    assert selected.candidate.identifier == candidate.identifier
    assert selected.selection_index == 2
    assert selected.score == score
    assert selected.reason_codes == ("high_quality", "stable_tie_break")
    assert selected.variant_group_id == "variant_test"
    assert selected.tie_break_applied is True
