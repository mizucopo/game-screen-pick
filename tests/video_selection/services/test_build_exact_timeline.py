"""exact video timeline構築のtest。"""

from fractions import Fraction

import pytest

from src.video_selection.models.media_stream import MediaStream
from src.video_selection.services.build_exact_timeline import build_exact_timeline


def _stream(
    *,
    start_pts: int | None = None,
    duration_ts: int | None = None,
) -> MediaStream:
    return MediaStream(
        index=0,
        kind="video",
        codec_name="ffv1",
        time_base=Fraction(1, 1000),
        start_pts=start_pts,
        duration_ts=duration_ts,
        width=64,
        height=48,
        sample_rate=None,
        channels=None,
        language=None,
        is_default=True,
        is_forced=False,
        is_attached_picture=False,
    )


def test_frame_duration_builds_gapless_timeline_with_later_boundary_ownership() -> None:
    """最終frame終端とscene境界からgapのないtimelineが構築されること。

    Arrange:
        - originが100で最終frame終端が1秒となるVFR timingが用意される
        - 0.25秒と0.75秒にscene signalが用意される
    Act:
        - exact timelineが構築される
    Assert:
        - 3区間が0から1秒をgapとoverlapなしで覆うこと
        - 0.25秒境界が後側区間に所属すること
    """
    # Arrange
    scene_pts = (350, 850, 1100)

    # Act
    timeline = build_exact_timeline(
        video_fingerprint="a" * 64,
        stream=_stream(),
        origin_pts=100,
        last_frame_pts=850,
        last_frame_duration_ts=250,
        scene_pts=scene_pts,
    )

    # Assert
    assert timeline.duration.seconds == Fraction(1)
    assert [(item.start, item.end) for item in timeline.segments] == [
        (Fraction(0), Fraction(1, 4)),
        (Fraction(1, 4), Fraction(3, 4)),
        (Fraction(3, 4), Fraction(1)),
    ]
    assert timeline.segment_at(Fraction(1, 4)) == timeline.segments[1]
    assert all(item.identifier.startswith("seg_") for item in timeline.segments)
    assert all(len(item.identifier) == 68 for item in timeline.segments)


def test_stream_duration_is_used_only_when_last_frame_duration_is_missing() -> None:
    """最終frame duration欠落時だけstreamのexact終端が使われること。

    Arrange:
        - exactなstream startとdurationを持つtimingが用意される
    Act:
        - 最終frame durationなしでtimelineが構築される
    Assert:
        - stream終端からoriginを引いた3/4秒がVideo Durationになること
    """
    # Arrange
    stream = _stream(start_pts=100, duration_ts=1000)

    # Act
    timeline = build_exact_timeline(
        video_fingerprint="b" * 64,
        stream=stream,
        origin_pts=350,
        last_frame_pts=850,
        last_frame_duration_ts=None,
        scene_pts=(),
    )

    # Assert
    assert timeline.duration.seconds == Fraction(3, 4)


def test_timeline_without_exact_positive_end_fails_fast() -> None:
    """exactな正の終端を得られないtimelineが拒否されること。

    Arrange:
        - frame durationとstream durationの両方を欠くtimingが用意される
    Act:
        - timeline構築が試行される
    Assert:
        - floatやframe間隔で推測されず失敗すること
    """
    # Arrange
    stream = _stream()

    # Act
    with pytest.raises(ValueError) as error:
        build_exact_timeline(
            video_fingerprint="c" * 64,
            stream=stream,
            origin_pts=0,
            last_frame_pts=1000,
            last_frame_duration_ts=None,
            scene_pts=(),
        )

    # Assert
    assert "Video Duration" in str(error.value)
