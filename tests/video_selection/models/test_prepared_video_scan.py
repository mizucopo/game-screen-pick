"""PreparedVideoScanのtest。"""

from src.video_selection.models.prepared_video_scan import PreparedVideoScan


def test_scan_disposition_duration_and_speed_are_preserved() -> None:
    """先行確定されたscanの再利用状態、時間、速度が保持されること。

    Arrange:
        - cold scanの完了値が用意される
    Act:
        - PreparedVideoScanが構築される
    Assert:
        - 再利用状態、所要時間、入力秒毎wall秒が保持されること
    """
    # Arrange
    reused = False
    duration_seconds = 12.5
    input_seconds_per_wall_second = 1.75

    # Act
    result = PreparedVideoScan(
        reused=reused,
        duration_seconds=duration_seconds,
        input_seconds_per_wall_second=input_seconds_per_wall_second,
    )

    # Assert
    assert result.reused is False
    assert result.duration_seconds == 12.5
    assert result.input_seconds_per_wall_second == 1.75
