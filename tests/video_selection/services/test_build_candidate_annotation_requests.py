"""Candidate Annotation shortlist構築のtest。"""

from dataclasses import replace
from fractions import Fraction
from pathlib import Path

from src.video_selection.models.candidate_annotation_request import (
    CandidateAnnotationRequest,
)
from src.video_selection.models.candidate_moment import CandidateMoment
from src.video_selection.models.completed_stage import CompletedStage
from src.video_selection.models.content_reject_reason import ContentRejectReason
from src.video_selection.models.context_cue import ContextCue
from src.video_selection.models.context_cue_equivalence_group import (
    ContextCueEquivalenceGroup,
)
from src.video_selection.models.context_stage_result import ContextStageResult
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
from src.video_selection.models.stage_fingerprint import StageFingerprint
from src.video_selection.models.timeline_segment import TimelineSegment
from src.video_selection.models.video_duration import VideoDuration
from src.video_selection.models.video_scan_metrics import VideoScanMetrics
from src.video_selection.models.video_scan_result import VideoScanResult
from src.video_selection.models.video_source import VideoSource
from src.video_selection.models.video_stage_result import VideoStageResult
from src.video_selection.models.video_timeline import VideoTimeline
from src.video_selection.services.build_candidate_annotation_requests import (
    build_candidate_annotation_requests,
    select_scene_catalog_representatives,
)


def test_local_quality_representative_drives_shortlist_and_annotation_input(
    tmp_path: Path,
) -> None:
    """Primaryを先頭に同じMomentの代替frameも意味注釈候補へ保持されること。

    Arrange:
        - 最高品質frameを2枚持つMomentと、近似・多様な2 Momentが用意される
    Act:
        - Candidate Annotation requestとScene Catalog代表が構築される
    Assert:
        - 各Momentの最高品質frameがPrimaryとして先頭に使われること
        - 近似Momentより多様なMomentが先に並べられること
        - requestには同一Momentの独立fallback候補も品質順で保持されること
        - Scene CatalogにはPrimaryだけが渡されること
    """
    # Arrange
    first_low = _frame("a", quality=0.70, feature=(1.0, 0.0), second=1)
    first_high = _frame("b", quality=0.95, feature=(1.0, 0.0), second=2)
    near = _frame("c", quality=0.90, feature=(0.999, 0.001), second=3)
    diverse = _frame("d", quality=0.80, feature=(0.0, 1.0), second=4)
    result = _stage_result(
        tmp_path,
        "1",
        duration=10,
        moments=(
            _moment("1", Fraction(2), (first_low.identifier, first_high.identifier)),
            _moment("2", Fraction(3), (near.identifier,)),
            _moment("3", Fraction(4), (diverse.identifier,)),
        ),
        candidates=(first_low, first_high, near, diverse),
    )

    # Act
    requests = build_candidate_annotation_requests(
        (result,),
        selection_intent="ブログ本文を説明できる画像を選ぶ",
    )
    representatives = select_scene_catalog_representatives(requests)

    # Assert
    assert [request.moment.identifier for request in requests] == [
        "mom_" + "1" * 64,
        "mom_" + "3" * 64,
        "mom_" + "2" * 64,
    ]
    assert [frame.identifier for frame in representatives] == [
        first_high.identifier,
        diverse.identifier,
        near.identifier,
    ]
    assert requests[0].frame_candidates == (first_high, first_low)
    assert requests[0].moment.frame_candidate_ids == (
        first_high.identifier,
        first_low.identifier,
    )


def test_annotation_request_keeps_only_three_best_same_moment_frames(
    tmp_path: Path,
) -> None:
    """同じMomentの候補がPrimaryを含む品質上位3枚へ制限されること。

    Arrange:
        - 品質の異なる4枚の適格frameを持つ一つのMomentが用意される
    Act:
        - Candidate Annotation requestが構築される
    Assert:
        - 品質上位3枚だけが決定的な順序で保持されること
    """
    # Arrange
    frames = (
        _frame("a", quality=0.80, feature=(1.0, 0.0), second=1),
        _frame("b", quality=0.95, feature=(1.0, 0.0), second=2),
        _frame("c", quality=0.70, feature=(1.0, 0.0), second=3),
        _frame("d", quality=0.90, feature=(1.0, 0.0), second=4),
    )
    result = _stage_result(
        tmp_path,
        "1",
        duration=10,
        moments=(
            _moment(
                "1",
                Fraction(4),
                tuple(frame.identifier for frame in frames),
            ),
        ),
        candidates=frames,
    )

    # Act
    requests = build_candidate_annotation_requests(
        (result,),
        selection_intent="ブログ本文を説明できる画像を選ぶ",
    )

    # Assert
    assert requests[0].frame_candidates == (frames[1], frames[3], frames[0])
    assert requests[0].moment.frame_candidate_ids == tuple(
        frame.identifier for frame in requests[0].frame_candidates
    )


def test_ties_follow_video_order_time_moment_and_frame_id(tmp_path: Path) -> None:
    """同点がVideo Order、Video Time、Moment ID、Frame IDで固定されること。

    Arrange:
        - 同じ品質・特徴で時刻とIDだけが異なるMomentとframeが用意される
    Act:
        - Candidate Annotation requestが構築される
    Assert:
        - source順、時刻順、Moment ID順でshortlistが固定されること
        - local代表の同品質frameがFrame ID順で選ばれること
    """
    # Arrange
    later_frame = _frame("e", quality=0.8, feature=(1.0, 0.0), second=8)
    earlier_high_id = _frame("f", quality=0.8, feature=(1.0, 0.0), second=3)
    earlier_low_id = _frame("a", quality=0.8, feature=(1.0, 0.0), second=3)
    second_video_frame = replace(
        _frame("b", quality=0.8, feature=(1.0, 0.0), second=1),
        video_fingerprint="2" * 64,
    )
    first = _stage_result(
        tmp_path,
        "1",
        duration=10,
        moments=(
            _moment("3", Fraction(8), (later_frame.identifier,)),
            _moment(
                "2",
                Fraction(3),
                (earlier_high_id.identifier, earlier_low_id.identifier),
            ),
        ),
        candidates=(later_frame, earlier_high_id, earlier_low_id),
    )
    second = _stage_result(
        tmp_path,
        "2",
        duration=10,
        moments=(_moment("1", Fraction(1), (second_video_frame.identifier,)),),
        candidates=(second_video_frame,),
    )

    # Act
    requests = build_candidate_annotation_requests(
        (first, second),
        selection_intent="ブログ本文を説明できる画像を選ぶ",
    )
    representatives = select_scene_catalog_representatives(requests)

    # Assert
    assert [request.moment.identifier for request in requests] == [
        "mom_" + "2" * 64,
        "mom_" + "3" * 64,
        "mom_" + "1" * 64,
    ]
    assert representatives[0].identifier == earlier_low_id.identifier


def test_scene_catalog_representatives_skip_shared_frames_and_fill_limit() -> None:
    """共有Frameが除外され後続の一意な代表で上限まで補充されること。

    Arrange:
        - 最初の2 Momentが同じFrame Candidateを共有し、後続Momentが別frameを持つ
    Act:
        - 上限2件のScene Catalog Representative Setが構築される
    Assert:
        - 重複Frame Candidate IDが除外され、後続frameで2件まで補充されること
    """
    # Arrange
    shared = _frame("a", quality=0.90, feature=(1.0, 0.0), second=1)
    later = _frame("b", quality=0.80, feature=(0.0, 1.0), second=3)
    requests = tuple(
        CandidateAnnotationRequest(
            moment=_moment(digest, Fraction(second), (frame.identifier,)),
            frame_candidates=(frame,),
            context_cues=(),
            video_set_progress=Fraction(second, 10),
            selection_intent="ブログ本文を説明できる画像を選ぶ",
            cue_selection_policy_version="nearby-context-v1",
        )
        for digest, second, frame in (
            ("1", 1, shared),
            ("2", 2, shared),
            ("3", 3, later),
        )
    )

    # Act
    representatives = select_scene_catalog_representatives(requests, limit=2)

    # Assert
    assert [frame.identifier for frame in representatives] == [
        shared.identifier,
        later.identifier,
    ]


def test_annotation_shortlist_skips_shared_representative_frames(
    tmp_path: Path,
) -> None:
    """共有Frameが一度だけ注釈され後続の一意なMomentが保持されること。

    Arrange:
        - 最初の2 Momentが同じFrame Candidateを共有し、後続Momentが別frameを持つ
    Act:
        - Candidate Annotation requestが構築される
    Assert:
        - shortlist順で最初の共有Frameだけが保持されること
        - 後続の一意なFrameを持つMomentが失われないこと
    """
    # Arrange
    shared = _frame("a", quality=0.90, feature=(1.0, 0.0), second=1)
    later = _frame("b", quality=0.80, feature=(0.0, 1.0), second=3)
    result = _stage_result(
        tmp_path,
        "1",
        duration=10,
        moments=(
            _moment("1", Fraction(1), (shared.identifier,)),
            _moment("2", Fraction(2), (shared.identifier,)),
            _moment("3", Fraction(3), (later.identifier,)),
        ),
        candidates=(shared, later),
    )

    # Act
    requests = build_candidate_annotation_requests(
        (result,),
        selection_intent="ブログ本文を説明できる画像を選ぶ",
    )

    # Assert
    assert [request.moment.identifier for request in requests] == [
        "mom_" + "1" * 64,
        "mom_" + "3" * 64,
    ]
    assert [request.frame_candidates[0].identifier for request in requests] == [
        shared.identifier,
        later.identifier,
    ]


def test_context_uses_nearby_equivalence_representatives_and_global_progress(
    tmp_path: Path,
) -> None:
    """近傍の代表Cueだけが距離選抜されVideo Set進行率が計算されること。

    Arrange:
        - 近傍Cue、同内容の非代表Cue、遠方Cueを持つ2本目のMomentが用意される
    Act:
        - Candidate Annotation requestが構築される
    Assert:
        - anchor前後15秒と重なる代表Cueのうち距離上位3件だけが時系列で渡されること
        - 1本目のdurationを含むVideo Set進行率が設定されること
        - nearby-context-v1がfingerprint入力用policyとして設定されること
    """
    # Arrange
    first_frame = _frame("1", quality=0.9, feature=(1.0, 0.0), second=1)
    second_frame = replace(
        _frame("2", quality=0.8, feature=(0.0, 1.0), second=20),
        video_fingerprint="2" * 64,
    )
    first = _stage_result(
        tmp_path,
        "1",
        duration=10,
        moments=(_moment("1", Fraction(1), (first_frame.identifier,)),),
        candidates=(first_frame,),
    )
    cues = (
        _cue("a", "2", 4, 6),
        _cue("b", "2", 19, 21),
        _cue("c", "2", 22, 23),
        _cue("d", "2", 34, 36),
        _cue("e", "2", 10, 11),
        _cue("f", "2", 40, 41),
    )
    second = _stage_result(
        tmp_path,
        "2",
        duration=30,
        moments=(_moment("2", Fraction(20), (second_frame.identifier,)),),
        candidates=(second_frame,),
        cues=cues,
        equivalence_groups=(
            ContextCueEquivalenceGroup(
                representative_cue_id=cues[1].identifier,
                cue_ids=(cues[1].identifier, cues[2].identifier),
            ),
        ),
    )

    # Act
    requests = build_candidate_annotation_requests(
        (first, second),
        selection_intent="ブログ本文を説明できる画像を選ぶ",
    )
    target = next(
        request
        for request in requests
        if request.moment.identifier == "mom_" + "2" * 64
    )

    # Assert
    assert [cue.identifier for cue in target.context_cues] == [
        cues[0].identifier,
        cues[4].identifier,
        cues[1].identifier,
    ]
    assert target.video_set_progress == Fraction(3, 4)
    assert target.cue_selection_policy_version == "nearby-context-v1"


def _frame(
    digest: str,
    *,
    quality: float,
    feature: tuple[float, ...],
    second: int,
) -> FrameCandidate:
    """適格なFrame Candidateを構築する。"""
    return FrameCandidate(
        identifier="frm_" + digest * 64,
        image_bytes=("image-" + digest).encode(),
        video_fingerprint="1" * 64,
        stream_index=0,
        source_pts=second,
        origin_pts=0,
        time_base=Fraction(1),
        video_time=Fraction(second),
        analysis=NeutralImageAnalysis(
            source_pts=second,
            metrics=_metrics(),
            quality_score=quality,
            visual_feature=feature,
            grayscale_signature=digest.encode(),
            reject_reason=None,
        ),
    )


def _moment(
    digest: str,
    second: Fraction,
    frame_ids: tuple[str, ...],
) -> CandidateMoment:
    """Frame参照を持つCandidate Momentを構築する。"""
    return CandidateMoment(
        identifier="mom_" + digest * 64,
        source_pts=int(second),
        anchor_time=second,
        timeline_segment_id="seg_" + digest * 64,
        evidence=("heartbeat",),
        proxy_quality_score=0.8,
        frame_candidate_ids=frame_ids,
    )


def _cue(
    digest: str,
    video_digest: str,
    start: int,
    end: int,
) -> ContextCue:
    """usableなContext Cueを構築する。"""
    return ContextCue(
        identifier="cue_" + digest * 64,
        video_fingerprint=video_digest * 64,
        start=Fraction(start),
        end=Fraction(end),
        text="context " + digest,
    )


def _stage_result(
    tmp_path: Path,
    digest: str,
    *,
    duration: int,
    moments: tuple[CandidateMoment, ...],
    candidates: tuple[FrameCandidate, ...],
    cues: tuple[ContextCue, ...] = (),
    equivalence_groups: tuple[ContextCueEquivalenceGroup, ...] = (),
) -> VideoStageResult:
    """shortlist構築に必要なVideo Stage resultを構築する。"""
    timeline = VideoTimeline(
        origin_pts=0,
        time_base=Fraction(1),
        duration=VideoDuration(Fraction(duration)),
        segments=(
            TimelineSegment(
                identifier="seg_" + digest * 64,
                start=Fraction(0),
                end=Fraction(duration),
            ),
        ),
    )
    primary_stream = MediaStream(
        index=0,
        kind="video",
        codec_name="ffv1",
        time_base=Fraction(1),
        start_pts=0,
        duration_ts=duration,
        width=64,
        height=48,
        sample_rate=None,
        channels=None,
        language=None,
        is_default=True,
        is_forced=False,
        is_attached_picture=False,
    )
    breakdown = ContentRejectReason.empty_breakdown()
    extraction = FrameCandidateExtraction(
        moments=moments,
        candidates=candidates,
        native_frame_count=len(candidates),
        reject_breakdown=breakdown,
        deduplicated_frame_count=0,
        zero_frame_moment_count=sum(not item.frame_candidate_ids for item in moments),
    )
    source = VideoSource(
        path=tmp_path / f"video-{digest}.mkv",
        relative_path=f"video-{digest}.mkv",
        fingerprint=digest * 64,
        size_bytes=1,
        modified_at_ns=1,
    )
    return VideoStageResult(
        source=source,
        scan=VideoScanResult(
            primary_stream=primary_stream,
            timeline=timeline,
            heartbeats=(),
            scene_signals=(),
            metrics=VideoScanMetrics(
                input_duration=Fraction(duration),
                wall_seconds=1.0,
                cpu_seconds=0.5,
                input_seconds_per_wall_second=float(duration),
                decode_backend="cpu",
                decode_pass_count=1,
                heartbeat_count=0,
                heartbeat_bytes=0,
                heartbeat_max_gap_seconds=0.0,
                heartbeat_p95_gap_seconds=0.0,
                scene_signal_count=0,
                timeline_segment_count=1,
            ),
        ),
        extraction=extraction,
        extraction_metrics=FrameCandidateExtractionMetrics(
            wall_seconds=1.0,
            cpu_seconds=0.5,
            density_cap=len(moments),
            actual_moment_count=len(moments),
            native_frame_count=len(candidates),
            reject_breakdown=breakdown,
            deduplicated_frame_count=0,
            zero_frame_moment_count=sum(
                not item.frame_candidate_ids for item in moments
            ),
            frame_candidate_count=len(candidates),
            frame_candidate_bytes=sum(len(item.image_bytes) for item in candidates),
        ),
        context=ContextStageResult(
            cues=cues,
            outcomes=(),
            equivalence_groups=equivalence_groups,
        ),
        completed_stages=(
            CompletedStage(ProcessingStage.SCAN_VIDEO, StageFingerprint("a" * 64)),
            CompletedStage(
                ProcessingStage.EXTRACT_FRAME_CANDIDATES,
                StageFingerprint(digest * 64),
            ),
            CompletedStage(
                ProcessingStage.COLLECT_CONTEXT,
                StageFingerprint("c" * 64),
            ),
        ),
    )


def _metrics() -> NeutralImageMetrics:
    """適格判定用のNeutral Image Metricsを構築する。"""
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
