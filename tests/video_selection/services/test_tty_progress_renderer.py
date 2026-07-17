from src.video_selection.models.progress_event import ProgressEvent
from src.video_selection.services.tty_progress_renderer import TtyProgressRenderer


def test_tty_progress_renderer_updates_line_until_terminal_event() -> None:
    """TTY進捗が同じ行を更新しterminal eventで改行されること。

    Arrange:
        - 進行中eventとrun完了eventが用意される
    Act:
        - TTY rendererで両eventが描画される
    Assert:
        - 進行中は改行せず、完了時だけ改行されること
    """
    # Arrange
    renderer = TtyProgressRenderer()
    progress = ProgressEvent(
        kind="progress",
        severity="info",
        processed_count=1,
        total_count=2,
        reason_code="stage_progress",
    )
    completed = ProgressEvent(
        kind="run_completed",
        severity="info",
        reason_code="run_completed",
    )

    # Act
    progress_text = renderer.render(progress)
    completed_text = renderer.render(completed)

    # Assert
    assert (progress_text, completed_text) == (
        "\r[info] event=progress reason=stage_progress progress=1/2\x1b[K",
        "\r[info] event=run_completed reason=run_completed\x1b[K\n",
    )
