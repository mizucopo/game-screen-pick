"""NativeVideoScanのpartition対応test。"""

from fractions import Fraction

import pytest

from src.video_selection.models.native_video_scan import NativeVideoScan


def test_partition_without_owned_signals_is_valid() -> None:
    """正当なpartitionでheartbeatとsceneが0件でも保持されること。

    Arrange:
        - timeline frameはあるがsignal ownershipがないpartitionが用意される
    Act:
        - Native Video Scanが構築される
    Assert:
        - 空signal列を持つ1 decode結果として受理されること
    """
    # Arrange
    heartbeats = ()

    # Act
    scan = NativeVideoScan(
        stream_index=0,
        origin_pts=10,
        last_frame_pts=19,
        last_frame_duration_ts=1,
        time_base=Fraction(1, 10),
        heartbeats=heartbeats,
        scene_frames=(),
        wall_seconds=1.0,
        cpu_seconds=0.5,
        decode_pass_count=1,
    )

    # Assert
    assert scan.heartbeats == ()
    assert scan.scene_frames == ()


def test_partition_with_reversed_timeline_is_rejected() -> None:
    """終端が原点より前のpartitionがdomain不正として拒否されること。

    Arrange:
        - 原点20に対して最終frame 10のpartition値が用意される
    Act:
        - Native Video Scanの構築が試行される
    Assert:
        - 不正なtimingとして拒否されること
    """
    # Arrange
    origin_pts = 20
    last_frame_pts = 10

    # Act
    with pytest.raises(ValueError) as error:
        NativeVideoScan(
            stream_index=0,
            origin_pts=origin_pts,
            last_frame_pts=last_frame_pts,
            last_frame_duration_ts=1,
            time_base=Fraction(1, 10),
            heartbeats=(),
            scene_frames=(),
            wall_seconds=1.0,
            cpu_seconds=0.5,
            decode_pass_count=1,
        )

    # Assert
    assert "timingまたはmetric" in str(error.value)


def test_incomplete_frame_timing_hint_is_rejected() -> None:
    """片方だけのframe timing resource hintが拒否されること。

    Arrange:
        - 最小PTS差だけを持つpartition値が用意される
    Act:
        - Native Video Scanの構築が試行される
    Assert:
        - 不正なtimingとして拒否されること
    """
    # Arrange
    minimum_frame_delta_ts = 1

    # Act
    with pytest.raises(ValueError) as error:
        NativeVideoScan(
            stream_index=0,
            origin_pts=0,
            last_frame_pts=10,
            last_frame_duration_ts=1,
            time_base=Fraction(1, 10),
            heartbeats=(),
            scene_frames=(),
            wall_seconds=1.0,
            cpu_seconds=0.5,
            decode_pass_count=1,
            minimum_frame_delta_ts=minimum_frame_delta_ts,
            maximum_frame_count_per_pts=None,
        )

    # Assert
    assert "timingまたはmetric" in str(error.value)
