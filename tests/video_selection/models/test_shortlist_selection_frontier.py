"""ShortlistSelectionFrontierの契約test。"""

import pytest

from src.video_selection.models.shortlist_selection_frontier import (
    ShortlistSelectionFrontier,
)
from src.video_selection.models.stage_fingerprint import StageFingerprint


def test_frontier_owns_checkpoint_identity_and_incomplete_artifact() -> None:
    """不足境界からWork Unit identityと選択結果を含まないartifactが導出されること。

    Arrange:
        - 選定意味入力fingerprintと累積Candidate件数が用意される
    Act:
        - Shortlist Selection Frontierが構築される
    Assert:
        - key、意味入力、不足証明artifactが一体で返されること
    """
    # Arrange
    request_fingerprint = StageFingerprint("a" * 64)

    # Act
    frontier = ShortlistSelectionFrontier(
        selection_request_fingerprint=request_fingerprint,
        annotated_candidate_count=24,
    )

    # Assert
    assert frontier.work_unit_key == "annotated-candidate-count-24"
    assert frontier.semantic_input == {
        "selection_request_fingerprint": request_fingerprint.value,
        "annotated_candidate_count": 24,
    }
    assert frontier.artifact == {
        "schema": "game-screen-pick/shortlist-selection-frontier@1.0.0",
        "selection_request_fingerprint": request_fingerprint.value,
        "annotated_candidate_count": 24,
        "selection_can_stop": False,
    }


def test_non_positive_frontier_boundary_is_rejected() -> None:
    """正でない累積Candidate件数が拒否されること。

    Arrange:
        - 0件のFrontier入力が用意される
    Act:
        - Shortlist Selection Frontierが構築される
    Assert:
        - ValueErrorが送出されること
    """
    # Arrange
    candidate_count = 0

    # Act
    with pytest.raises(ValueError) as captured:
        ShortlistSelectionFrontier(
            selection_request_fingerprint=StageFingerprint("b" * 64),
            annotated_candidate_count=candidate_count,
        )

    # Assert
    assert "1以上" in str(captured.value)
