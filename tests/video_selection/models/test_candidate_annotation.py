from src.video_selection.models.candidate_annotation import CandidateAnnotation
from src.video_selection.models.frame_candidate import FrameCandidate


def test_major_spoiler_requires_safe_evidence_summary() -> None:
    """high Spoiler Riskに引用ではないevidence summaryが要求されること。

    Arrange:
        - high riskと短い意味証拠を持つCandidate Annotationが用意される
    Act:
        - Candidate Annotationが構築される
    Assert:
        - riskとevidenceが分離して保持されること
    """
    # Arrange
    candidate = FrameCandidate(identifier="frame-1", image_bytes=b"image")

    # Act
    annotation = CandidateAnnotation(
        candidate=candidate,
        summary="終盤の対決場面",
        candidate_moment_id="mom_" + "a" * 64,
        scene_slug="climax",
        blog_image_type="event",
        explanation_value="high",
        frame_choice_reason="対決する人物が明確に写る",
        screen_text_kind="dialogue",
        context_relevance="strong",
        supporting_context_cue_ids=("cue-1",),
        spoiler_risk="high",
        spoiler_evidence="主要人物の正体が明示される",
    )

    # Assert
    assert annotation.candidate is candidate
    assert annotation.spoiler_risk == "high"
    assert annotation.spoiler_evidence == "主要人物の正体が明示される"
    assert not hasattr(annotation, "quality_score")
    assert not hasattr(annotation, "final_score")
    assert not hasattr(annotation, "selected")
