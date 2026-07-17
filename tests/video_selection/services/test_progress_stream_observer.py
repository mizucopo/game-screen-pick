from io import StringIO

import pytest

from src.video_selection.models.progress_event import ProgressEvent
from src.video_selection.services.progress_stream_observer import (
    ProgressStreamObserver,
)


@pytest.mark.parametrize(
    ("is_tty", "expected"),
    [
        (
            False,
            "[info] event=progress reason=stage_progress progress=1/2\n",
        ),
        (
            True,
            "\r[info] event=progress reason=stage_progress progress=1/2\x1b[K",
        ),
    ],
)
def test_progress_stream_observer_selects_renderer_from_stream(
    monkeypatch: pytest.MonkeyPatch,
    is_tty: bool,
    expected: str,
) -> None:
    """streamのTTY状態から対応rendererが自動選択されること。

    Arrange:
        - TTY状態を制御したmemory text streamが用意される
    Act:
        - Progress Eventがstream observerへ通知される
    Assert:
        - TTY状態に対応した形式だけがstreamへ書かれること
    """
    # Arrange
    stream = StringIO()
    monkeypatch.setattr(stream, "isatty", lambda: is_tty)
    observer = ProgressStreamObserver(stream)
    event = ProgressEvent(
        kind="progress",
        severity="info",
        processed_count=1,
        total_count=2,
        reason_code="stage_progress",
    )

    # Act
    observer.observe(event)

    # Assert
    assert stream.getvalue() == expected
