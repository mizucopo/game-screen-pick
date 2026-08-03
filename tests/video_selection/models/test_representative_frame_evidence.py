"""Representative Frame Evidenceのtest。"""

from src.video_selection.models.representative_frame_evidence import (
    CandidateFrameContentKind,
    RepresentativeFrameEvidence,
)


def test_representative_frame_evidence_keeps_selection_observations() -> None:
    """Representative Frame比較用の構造化観測が保持されること。

    Arrange:
        - 戦闘action、明瞭な主対象と敵、遮蔽なしの観測値が用意される
    Act:
        - Representative Frame Evidenceが構築される
    Assert:
        - 比較に必要な観測値が変更されず保持されること
    """
    # Arrange
    content_kind: CandidateFrameContentKind = "gameplay_action"

    # Act
    evidence = RepresentativeFrameEvidence(
        content_kind=content_kind,
        primary_subject_visibility="clear",
        opponent_body_visibility="clear",
        transient_obstruction="none",
    )

    # Assert
    assert (
        evidence.content_kind,
        evidence.primary_subject_visibility,
        evidence.opponent_body_visibility,
        evidence.transient_obstruction,
    ) == (content_kind, "clear", "clear", "none")
