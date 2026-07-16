"""Blog Candidate domain contractのtest。"""

from fractions import Fraction

import pytest

from src.video_selection.models.blog_candidate import BlogCandidate
from src.video_selection.models.candidate_annotation import CandidateAnnotation
from src.video_selection.models.content_reject_reason import ContentRejectReason
from src.video_selection.models.frame_candidate import FrameCandidate
from src.video_selection.models.neutral_image_analysis import NeutralImageAnalysis
from src.video_selection.models.neutral_image_metrics import NeutralImageMetrics


def test_context_cue_cannot_make_rejected_frame_eligible() -> None:
    """strong Context Cueでも不適格frameがBlog Candidateにされないこと。

    Arrange:
        - blackoutとしてrejectされたFrame Candidateが用意される
        - strong Context Cue Relevanceを持つCandidate Annotationが用意される
    Act:
        - Blog Candidateが構築される
    Assert:
        - Context Cueによる適格化が拒否されること
    """
    # Arrange
    metrics = NeutralImageMetrics(
        blur_score=0.0,
        brightness=0.0,
        contrast=0.0,
        edge_density=0.0,
        color_richness=0.0,
        ui_density=0.0,
        action_intensity=0.0,
        visual_balance=0.0,
        dramatic_score=0.0,
        luminance_entropy=0.0,
        luminance_range=0.0,
        near_black_ratio=1.0,
        near_white_ratio=0.0,
        dominant_tone_ratio=1.0,
        information_score=0.0,
        visibility_score=0.0,
    )
    frame = FrameCandidate(
        identifier="frm_" + "a" * 64,
        image_bytes=b"blackout",
        video_fingerprint="a" * 64,
        stream_index=0,
        source_pts=0,
        origin_pts=0,
        time_base=Fraction(1, 1000),
        video_time=Fraction(0),
        analysis=NeutralImageAnalysis(
            source_pts=0,
            metrics=metrics,
            quality_score=0.0,
            visual_feature=(1.0, 0.0),
            grayscale_signature=b"black",
            reject_reason=ContentRejectReason.BLACKOUT,
        ),
    )
    annotation = CandidateAnnotation(
        candidate=frame,
        candidate_moment_id="mom_" + "a" * 64,
        summary="暗転frame",
        context_relevance="strong",
        supporting_context_cue_ids=("cue_" + "a" * 64,),
    )

    # Act
    # Assert
    with pytest.raises(ValueError, match="有効なframe"):
        BlogCandidate(
            annotation=annotation,
            scene_selection_role="ordinary",
            video_order=0,
            video_set_progress=Fraction(0),
            shortlist_rank=0,
        )
