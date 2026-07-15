"""Frame Refinementのtest。"""

from fractions import Fraction

import numpy as np

from src.video_selection.models.candidate_moment import CandidateMoment
from src.video_selection.models.content_reject_reason import ContentRejectReason
from src.video_selection.models.decoded_video_frame import DecodedVideoFrame
from src.video_selection.models.timeline_segment import TimelineSegment
from src.video_selection.models.video_duration import VideoDuration
from src.video_selection.models.video_timeline import VideoTimeline
from src.video_selection.services.refine_candidate_moments import (
    refine_candidate_moments,
)


def _timeline(duration_seconds: int = 5) -> VideoTimeline:
    return VideoTimeline(
        origin_pts=0,
        time_base=Fraction(1, 10),
        duration=VideoDuration(Fraction(duration_seconds)),
        segments=(
            TimelineSegment(
                "seg_" + "1" * 64,
                Fraction(0),
                Fraction(duration_seconds),
            ),
        ),
    )


def _moment(digest_character: str, second: Fraction) -> CandidateMoment:
    return CandidateMoment(
        identifier="mom_" + digest_character * 64,
        source_pts=int(second * 10),
        anchor_time=second,
        timeline_segment_id="seg_" + "1" * 64,
        evidence=("heartbeat",),
        proxy_quality_score=0.8,
    )


def _detailed_frame(source_pts: int, shift: int = 0) -> DecodedVideoFrame:
    rows, columns = np.indices((48, 64))
    values = ((rows // 3 + columns // 4 + shift) % 3 * 90 + 25).astype(np.uint8)
    rgb = np.stack(
        (values, np.roll(values, shift + 3, axis=1), 255 - values),
        axis=2,
    )
    return DecodedVideoFrame(
        stream_index=0,
        pts=source_pts,
        duration_ts=1,
        time_base=Fraction(1, 10),
        width=64,
        height=48,
        pixel_format="rgb24",
        pixels=rgb.tobytes(),
    )


def test_overlapping_moments_share_same_pts_frame_candidate() -> None:
    """重なるMomentから同一PTSのFrame Candidateが共有されること。

    Arrange:
        - refinement範囲が重なる2つのMomentと3つのnative frameが用意される
    Act:
        - 各Momentから最大2frameが選抜される
    Assert:
        - 両Momentが少なくとも一つの同じFrame Candidate IDを参照すること
        - Video Source内のFrame Candidate IDが重複生成されないこと
    """
    # Arrange
    moments = (
        _moment("2", Fraction(2)),
        _moment("3", Fraction(3)),
    )
    frames = (
        _detailed_frame(10, 0),
        _detailed_frame(20, 1),
        _detailed_frame(30, 2),
    )

    # Act
    extraction = refine_candidate_moments(
        video_fingerprint="a" * 64,
        timeline=_timeline(),
        moments=moments,
        frames=frames,
        refinement_radius_seconds=1.1,
        max_frame_candidates=2,
    )

    # Assert
    first_ids = set(extraction.moments[0].frame_candidate_ids)
    second_ids = set(extraction.moments[1].frame_candidate_ids)
    assert first_ids & second_ids
    assert len({item.identifier for item in extraction.candidates}) == len(
        extraction.candidates
    )
    assert all(item.identifier.startswith("frm_") for item in extraction.candidates)


def test_per_moment_deduplication_and_zero_frame_moment_are_reported() -> None:
    """近似重複がMoment内だけで除外され0-frame Momentが保持されること。

    Arrange:
        - 同一画面に近い2frame、黒frame、frameを持たないMomentが用意される
    Act:
        - Frame Refinementが実行される
    Assert:
        - 近似重複の片方だけが最初のMomentへ残ること
        - 黒frameがstable reasonへ集計されること
        - 最後のMomentが0-frameのまま診断対象として保持されること
    """
    # Arrange
    detailed = _detailed_frame(10, 0)
    almost_same_pixels = np.frombuffer(detailed.pixels, dtype=np.uint8).copy()
    almost_same_pixels[::97] = np.minimum(almost_same_pixels[::97] + 1, 255)
    almost_same = DecodedVideoFrame(
        stream_index=0,
        pts=11,
        duration_ts=1,
        time_base=Fraction(1, 10),
        width=64,
        height=48,
        pixel_format="rgb24",
        pixels=almost_same_pixels.tobytes(),
    )
    black = DecodedVideoFrame(
        stream_index=0,
        pts=40,
        duration_ts=1,
        time_base=Fraction(1, 10),
        width=64,
        height=48,
        pixel_format="rgb24",
        pixels=bytes(64 * 48 * 3),
    )
    moments = (
        _moment("4", Fraction(1)),
        _moment("5", Fraction(4)),
    )

    # Act
    extraction = refine_candidate_moments(
        video_fingerprint="b" * 64,
        timeline=_timeline(),
        moments=moments,
        frames=(detailed, almost_same, black),
        refinement_radius_seconds=0.2,
        max_frame_candidates=3,
    )

    # Assert
    assert len(extraction.moments[0].frame_candidate_ids) == 1
    assert extraction.moments[1].frame_candidate_ids == ()
    assert extraction.deduplicated_frame_count == 1
    assert extraction.zero_frame_moment_count == 1
    assert extraction.reject_breakdown[ContentRejectReason.BLACKOUT.value] == 1
