"""Video Set selection model test用の実体fixture。"""

from fractions import Fraction

from src.video_selection.models.blog_candidate import BlogCandidate
from src.video_selection.models.candidate_annotation import (
    CandidateAnnotation,
    SpoilerRisk,
)
from src.video_selection.models.frame_candidate import FrameCandidate
from src.video_selection.models.neutral_image_analysis import NeutralImageAnalysis
from src.video_selection.models.neutral_image_metrics import NeutralImageMetrics
from src.video_selection.models.selection_score import SelectionScore


def build_blog_candidate(
    digest_character: str = "a",
    *,
    spoiler_risk: SpoilerRisk = "none",
) -> BlogCandidate:
    """選定model test用の適格なBlog Candidateを構築する。"""
    metrics = NeutralImageMetrics(
        blur_score=100.0,
        brightness=100.0,
        contrast=50.0,
        edge_density=0.2,
        color_richness=0.5,
        ui_density=0.2,
        action_intensity=0.4,
        visual_balance=0.8,
        dramatic_score=0.3,
        luminance_entropy=1.0,
        luminance_range=100.0,
        near_black_ratio=0.0,
        near_white_ratio=0.0,
        dominant_tone_ratio=0.2,
        information_score=0.8,
        visibility_score=0.9,
    )
    frame = FrameCandidate(
        identifier="frm_" + digest_character * 64,
        image_bytes=digest_character.encode(),
        video_fingerprint=digest_character * 64,
        stream_index=0,
        source_pts=100,
        origin_pts=0,
        time_base=Fraction(1, 1000),
        video_time=Fraction(1, 10),
        analysis=NeutralImageAnalysis(
            source_pts=100,
            metrics=metrics,
            quality_score=0.8,
            visual_feature=(1.0, 0.0),
            grayscale_signature=b"signature",
            reject_reason=None,
        ),
    )
    annotation = CandidateAnnotation(
        candidate=frame,
        candidate_moment_id="mom_" + digest_character * 64,
        summary="選定model test candidate",
        scene_slug="test-scene",
        blog_image_type="normal_gameplay",
        explanation_value="high",
        context_relevance="unavailable",
        spoiler_risk=spoiler_risk,
        spoiler_evidence=("物語情報が画像に示される" if spoiler_risk != "none" else ""),
    )
    return BlogCandidate(
        annotation=annotation,
        scene_selection_role="ordinary",
        video_order=0,
        video_set_progress=Fraction(1, 10),
        shortlist_rank=1,
    )


def build_selection_score() -> SelectionScore:
    """選定判断を再現できる固定Selection Scoreを構築する。"""
    return SelectionScore(
        base_utility=0.81,
        spoiler_penalty=0.1,
        coverage_bonus=0.05,
        temporal_diversity_penalty=0.02,
        marginal_utility=0.74,
        similarity_pass=0.78,
        nearest_selected_similarity=0.75,
    )
