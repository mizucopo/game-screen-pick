"""Canonical publication test用の実体fixture。"""

from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

from src.video_selection.models.blog_candidate import BlogCandidate
from src.video_selection.models.candidate_annotation import CandidateAnnotation
from src.video_selection.models.candidate_moment import CandidateMoment
from src.video_selection.models.canonical_publication_request import (
    CanonicalPublicationRequest,
)
from src.video_selection.models.completed_stage import CompletedStage
from src.video_selection.models.content_reject_reason import ContentRejectReason
from src.video_selection.models.context_cue import ContextCue
from src.video_selection.models.context_cue_provenance import ContextCueProvenance
from src.video_selection.models.context_stage_result import ContextStageResult
from src.video_selection.models.effective_configuration import EffectiveConfiguration
from src.video_selection.models.frame_candidate import FrameCandidate
from src.video_selection.models.frame_candidate_extraction import (
    FrameCandidateExtraction,
)
from src.video_selection.models.frame_candidate_extraction_metrics import (
    FrameCandidateExtractionMetrics,
)
from src.video_selection.models.media_stream import MediaStream
from src.video_selection.models.neutral_image_analysis import NeutralImageAnalysis
from src.video_selection.models.neutral_image_metrics import NeutralImageMetrics
from src.video_selection.models.processing_stage import ProcessingStage
from src.video_selection.models.rejected_blog_candidate import RejectedBlogCandidate
from src.video_selection.models.report_provenance import ReportProvenance
from src.video_selection.models.report_stage_provenance import ReportStageProvenance
from src.video_selection.models.scene_catalog import SceneCatalog
from src.video_selection.models.scene_catalog_entry import SceneCatalogEntry
from src.video_selection.models.selected_blog_image import SelectedBlogImage
from src.video_selection.models.selection_rejection_reason import (
    SelectionRejectionReason,
)
from src.video_selection.models.selection_score import SelectionScore
from src.video_selection.models.stage_fingerprint import StageFingerprint
from src.video_selection.models.timeline_segment import TimelineSegment
from src.video_selection.models.video_duration import VideoDuration
from src.video_selection.models.video_scan_metrics import VideoScanMetrics
from src.video_selection.models.video_scan_result import VideoScanResult
from src.video_selection.models.video_set_selection_result import (
    VideoSetSelectionResult,
)
from src.video_selection.models.video_source import VideoSource
from src.video_selection.models.video_stage_result import VideoStageResult
from src.video_selection.models.video_timeline import VideoTimeline
from src.video_selection.services.discover_video_set import discover_video_set
from tests.video_selection.fakes.fake_model_runtime import FakeModelRuntime


def build_canonical_publication_request(
    tmp_path: Path,
    *,
    shortfall: bool = True,
    colliding_digest_prefixes: bool = False,
) -> CanonicalPublicationRequest:
    """完全なCanonical Publication Requestを構築する。"""
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    (input_folder / "chapter-01.mkv").write_bytes(b"canonical-video-content")
    video_set = discover_video_set(input_folder)
    source = video_set.sources[0]
    configuration = EffectiveConfiguration(
        video_input_folder=input_folder,
        output_folder=tmp_path / "output",
        image_count=2,
        spoiler_sensitivity="medium",
        similarity_threshold=0.72,
    )
    cue_id = "cue_" + "c" * 64
    first_digest = "123456789abc" + "a" * 52 if colliding_digest_prefixes else "a" * 64
    second_digest = "123456789abc" + "b" * 52 if colliding_digest_prefixes else "b" * 64
    first = _blog_candidate(
        source.fingerprint,
        first_digest,
        moment_digest="1" * 64,
        source_pts=15,
        video_time=Fraction(3, 2),
        shortlist_rank=1,
        context_cue_ids=(cue_id,),
        context_relevance="strong",
        summary="遺跡の広さとHUDが分かる通常play。",
    )
    second = _blog_candidate(
        source.fingerprint,
        second_digest,
        moment_digest="2" * 64,
        source_pts=30,
        video_time=Fraction(3),
        shortlist_rank=2,
        context_cue_ids=(),
        context_relevance="none",
        summary="似た構図の遺跡探索frame。",
    )
    selected = (
        _selected(first, index=1, marginal_utility=0.91),
        *(() if shortfall else (_selected(second, index=2, marginal_utility=0.82),)),
    )
    rejected = (
        (
            RejectedBlogCandidate(
                candidate=second,
                reason_code=SelectionRejectionReason.SIMILARITY_CEILING,
                counterfactual_score=_score(0.82),
                blocked_by_image_id=None,
                nearest_selected_image_id=first.identifier,
                similarity=0.985,
                variant_group_id="variant_" + "2" * 64,
            ),
        )
        if shortfall
        else ()
    )
    selection = VideoSetSelectionResult(
        selected=selected,
        rejected=rejected,
        requested_count=2,
        blog_image_type_targets={
            "normal_gameplay": 1,
            "event": 1,
            "menu": 0,
            "title": 0,
            "other": 0,
        },
        blog_image_type_actuals={
            "normal_gameplay": len(selected),
            "event": 0,
            "menu": 0,
            "title": 0,
            "other": 0,
        },
        final_similarity_ceiling=0.98 if shortfall else 0.72,
        major_spoiler_limit=None,
        annotated_candidate_count=2,
        shortlist_expansion_count=1,
        all_candidate_moments_exhausted=shortfall,
    )
    cue = ContextCue(
        identifier=cue_id,
        video_fingerprint=source.fingerprint,
        source_kind="embedded_subtitle",
        stream_index=1,
        start=Fraction(1),
        end=Fraction(2),
        timestamp_basis="source_pts",
        text="公開してはいけない秘密の台詞",
        language="ja",
        reliability="usable",
        provenance=ContextCueProvenance(
            codec_name="ass",
            source_pts=1000,
            source_time_base=Fraction(1, 1000),
            stream_language="ja",
            is_default=True,
            is_forced=False,
            language_source="stream_metadata",
        ),
    )
    video_stage = _video_stage_result(source, first, second, cue)
    models = FakeModelRuntime("canonical-publication").resolve_models(configuration)
    return CanonicalPublicationRequest(
        video_set=video_set,
        video_stage_results=(video_stage,),
        scene_catalog=_scene_catalog(),
        selection_result=selection,
        resolved_models=models,
        configuration=configuration,
        run_id="run_20260716T120000Z_fixture",
        started_at=datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc),
        completed_at=datetime(2026, 7, 16, 12, 1, tzinfo=timezone.utc),
        provenance=_provenance(),
    )


def _video_stage_result(
    source: VideoSource,
    first: BlogCandidate,
    second: BlogCandidate,
    cue: ContextCue,
) -> VideoStageResult:
    typed_source = first.annotation.candidate.video_fingerprint
    if (
        typed_source is None
        or typed_source != second.annotation.candidate.video_fingerprint
    ):
        raise AssertionError
    timeline = VideoTimeline(
        origin_pts=0,
        time_base=Fraction(1, 10),
        duration=VideoDuration(Fraction(10)),
        segments=(
            TimelineSegment(
                identifier="seg_" + "9" * 64,
                start=Fraction(0),
                end=Fraction(10),
            ),
        ),
    )
    primary_stream = MediaStream(
        index=0,
        kind="video",
        codec_name="ffv1",
        time_base=Fraction(1, 10),
        start_pts=0,
        duration_ts=100,
        width=64,
        height=48,
        sample_rate=None,
        channels=None,
        language=None,
        is_default=True,
        is_forced=False,
    )
    scan = VideoScanResult(
        primary_stream=primary_stream,
        timeline=timeline,
        heartbeats=(),
        scene_signals=(),
        metrics=VideoScanMetrics(
            input_duration=Fraction(10),
            wall_seconds=1.0,
            cpu_seconds=0.5,
            input_seconds_per_wall_second=10.0,
            decode_backend="cpu",
            decode_pass_count=1,
            heartbeat_count=0,
            heartbeat_bytes=0,
            heartbeat_max_gap_seconds=0.0,
            heartbeat_p95_gap_seconds=0.0,
            scene_signal_count=0,
            timeline_segment_count=1,
        ),
    )
    candidates = (first.annotation.candidate, second.annotation.candidate)
    moments = tuple(
        CandidateMoment(
            identifier=item.annotation.candidate_moment_id or "",
            source_pts=item.annotation.candidate.source_pts or 0,
            anchor_time=item.annotation.candidate.video_time or Fraction(0),
            timeline_segment_id=timeline.segments[0].identifier,
            evidence=("heartbeat",),
            proxy_quality_score=item.quality_score,
            frame_candidate_ids=(item.identifier,),
        )
        for item in (first, second)
    )
    breakdown = ContentRejectReason.empty_breakdown()
    extraction = FrameCandidateExtraction(
        moments=moments,
        candidates=candidates,
        native_frame_count=2,
        reject_breakdown=breakdown,
        deduplicated_frame_count=0,
        zero_frame_moment_count=0,
    )
    metrics = FrameCandidateExtractionMetrics(
        wall_seconds=0.5,
        cpu_seconds=0.25,
        density_cap=2,
        actual_moment_count=2,
        native_frame_count=2,
        reject_breakdown=breakdown,
        deduplicated_frame_count=0,
        zero_frame_moment_count=0,
        frame_candidate_count=2,
        frame_candidate_bytes=sum(len(item.image_bytes) for item in candidates),
    )
    return VideoStageResult(
        source=source,
        scan=scan,
        extraction=extraction,
        extraction_metrics=metrics,
        context=ContextStageResult(
            cues=(cue,),
            outcomes=(),
            completed_stage=None,
        ),
        completed_stages=(
            CompletedStage(ProcessingStage.SCAN_VIDEO, StageFingerprint("3" * 64)),
            CompletedStage(
                ProcessingStage.EXTRACT_FRAME_CANDIDATES,
                StageFingerprint("4" * 64),
            ),
            CompletedStage(
                ProcessingStage.COLLECT_CONTEXT,
                StageFingerprint("5" * 64),
            ),
        ),
    )


def _blog_candidate(
    video_fingerprint: str,
    frame_digest: str,
    *,
    moment_digest: str,
    source_pts: int,
    video_time: Fraction,
    shortlist_rank: int,
    context_cue_ids: tuple[str, ...],
    context_relevance: str,
    summary: str,
) -> BlogCandidate:
    frame = FrameCandidate(
        identifier="frm_" + frame_digest,
        image_bytes=b"proxy-image",
        video_fingerprint=video_fingerprint,
        stream_index=0,
        source_pts=source_pts,
        origin_pts=0,
        time_base=Fraction(1, 10),
        video_time=video_time,
        analysis=NeutralImageAnalysis(
            source_pts=source_pts,
            metrics=_neutral_metrics(),
            quality_score=0.9,
            visual_feature=(1.0, shortlist_rank / 10),
            grayscale_signature=b"signature",
            reject_reason=None,
        ),
    )
    annotation = CandidateAnnotation(
        candidate=frame,
        candidate_moment_id="mom_" + moment_digest,
        summary=summary,
        scene_slug="test-scene",
        blog_image_type="normal_gameplay",
        explanation_value="high",
        frame_choice_reason="構図と情報量が最も明瞭なframe。",
        screen_text_kind="hud",
        context_relevance=context_relevance,  # type: ignore[arg-type]
        supporting_context_cue_ids=context_cue_ids,
        spoiler_risk="none",
    )
    return BlogCandidate(
        annotation=annotation,
        scene_selection_role="recurring_gameplay",
        video_order=0,
        video_set_progress=video_time / 10,
        shortlist_rank=shortlist_rank,
    )


def _selected(
    candidate: BlogCandidate,
    *,
    index: int,
    marginal_utility: float,
) -> SelectedBlogImage:
    return SelectedBlogImage(
        candidate=candidate,
        selection_index=index,
        score=_score(marginal_utility),
        reason_codes=(
            "high_quality",
            "high_explanation_value",
            "normal_gameplay_coverage",
        ),
        variant_group_id="variant_" + str(index) * 64,
        tie_break_applied=False,
    )


def _score(marginal_utility: float) -> SelectionScore:
    return SelectionScore(
        base_utility=marginal_utility - 0.1,
        spoiler_penalty=0.0,
        coverage_bonus=0.1,
        temporal_diversity_penalty=0.0,
        marginal_utility=marginal_utility,
        similarity_pass=0.72,
        nearest_selected_similarity=None,
    )


def _scene_catalog() -> SceneCatalog:
    return SceneCatalog(
        (
            SceneCatalogEntry(
                slug="test-scene",
                display_name="探索",
                description="通常の探索画面",
                scene_kind="exploration",
                selection_role="recurring_gameplay",
            ),
            SceneCatalogEntry(
                slug="event",
                display_name="イベント",
                description="物語上のイベント画面",
                scene_kind="event",
                selection_role="cinematic",
            ),
            SceneCatalogEntry(
                slug="other",
                display_name="その他",
                description="他のsceneに分類されない画面",
                scene_kind="other",
                selection_role="ordinary",
            ),
        )
    )


def _provenance() -> ReportProvenance:
    stage = ReportStageProvenance(
        name="final_selection",
        fingerprint="stg_" + "6" * 64,
        upstream_fingerprints=(),
        cache_hits=0,
        cache_misses=1,
        recomputed_items=1,
        attempt_count=1,
        validation_failures=0,
        effective_settings={"requested_image_count": 2},
        tool_refs=(),
        model_refs=("candidate_annotation",),
        contract_refs=("video_set_selection_policy",),
        duration_ms=80,
    )
    return ReportProvenance(
        runtime={
            "os": "linux",
            "environment": "wsl2",
            "python": "3.13",
            "compute_device": "cuda",
            "gpu_model": "NVIDIA GeForce RTX 5090",
        },
        tools={
            "ffmpeg": "6.1.1",
            "ffprobe": "6.1.1",
            "ollama": "0.31.2",
            "faster_whisper": "1.2.1",
            "ctranslate2": "4.8.1",
        },
        contracts={
            "video_set_selection_policy": "video-set-selection-v4",
        },
        stages=(stage,),
    )


def _neutral_metrics() -> NeutralImageMetrics:
    return NeutralImageMetrics(
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
