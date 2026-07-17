from src.video_selection.models.processing_stage import ProcessingStage
from src.video_selection.models.progress_event import ProgressEvent
from src.video_selection.services.line_progress_renderer import LineProgressRenderer


def test_line_progress_renderer_escapes_control_characters() -> None:
    """relative pathの制御文字がescapeされ1 event 1行で描画されること。

    Arrange:
        - 改行を含む安全なrelative path付きProgress Eventが用意される
    Act:
        - line rendererでeventが描画される
    Assert:
        - 改行を増やさずstable fieldだけの一行が返されること
    """
    # Arrange
    event = ProgressEvent(
        kind="progress",
        severity="info",
        stage=ProcessingStage.SCAN_VIDEO,
        stage_index=1,
        stage_count=2,
        video_order=1,
        video_count=3,
        video_relative_path="chapter\n01.mkv",
        processed_count=5,
        total_count=10,
        cache_hit_count=1,
        cache_miss_count=4,
        reuse_count=1,
        recompute_count=4,
        elapsed_seconds=31.0,
        eta_seconds=45.0,
        estimation_state="available",
        work_unit_kind="video",
        reason_code="stage_progress",
    )
    renderer = LineProgressRenderer()

    # Act
    rendered = renderer.render(event)

    # Assert
    assert rendered == (
        "[info] event=progress reason=stage_progress stage=scan-video "
        'stage_index=1/2 video=1/3 path="chapter\\n01.mkv" '
        "progress=5/10 cache_hit=1 cache_miss=4 reuse=1 recompute=4 "
        "elapsed=31.0s eta=45.0s unit=video"
    )
