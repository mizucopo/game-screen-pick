"""partition非依存にScene Signalの最小間隔を適用する。"""

from fractions import Fraction

from ..models.scanned_video_frame import ScannedVideoFrame


def select_scene_signal_frames(
    frames: tuple[ScannedVideoFrame, ...],
    minimum_interval_seconds: float,
) -> tuple[ScannedVideoFrame, ...]:
    """PTS順の全scene候補へ一度だけgreedy intervalを適用する。"""
    interval = Fraction(str(minimum_interval_seconds))
    if interval <= 0:
        raise ValueError("Scene Signal intervalは正数である必要があります")
    selected: list[ScannedVideoFrame] = []
    previous_time: Fraction | None = None
    for frame in frames:
        current_time = frame.source_pts * frame.time_base
        if previous_time is not None and current_time <= previous_time:
            raise ValueError("Scene Signal candidateはPTS昇順である必要があります")
        if previous_time is None or current_time - previous_time >= interval:
            selected.append(frame)
            previous_time = current_time
    return tuple(selected)
