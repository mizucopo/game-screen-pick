import threading
from concurrent.futures import CancelledError
from types import TracebackType
from typing import Self

import pytest

from src.video_selection.vision.cancellable_json_requester import (
    CancellableJsonRequester,
)


class BlockingResponse:
    """closeされるまでJSON body読込を停止するHTTP response fake。"""

    def __init__(self) -> None:
        self.read_started = threading.Event()
        self.closed = threading.Event()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _exception_type: type[BaseException] | None,
        _exception: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()

    def read(self, _size: int = -1) -> bytes:
        self.read_started.set()
        if not self.closed.wait(timeout=1.0):
            raise RuntimeError("HTTP responseがcloseされませんでした")
        raise OSError("HTTP response closed")

    def close(self) -> None:
        self.closed.set()


def test_cancel_closes_active_http_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """中止要求でactive HTTP responseが閉じられrequestが終了すること。

    Arrange:
        - JSON body読込中で停止するHTTP responseが用意される
    Act:
        - 別threadのrequest中に中止が要求される
    Assert:
        - responseが閉じられ、requestがCancelledErrorで終了すること
    """
    # Arrange
    response = BlockingResponse()
    requester = CancellableJsonRequester()
    failures: list[BaseException] = []

    def open_response(_request: object, *, timeout: float) -> BlockingResponse:
        assert timeout == 60.0
        return response

    monkeypatch.setattr(
        "src.video_selection.vision.cancellable_json_requester.urlopen",
        open_response,
    )

    def request() -> None:
        try:
            requester(
                "POST",
                "http://localhost:11434/api/chat",
                {"model": "vision"},
                60.0,
            )
        except BaseException as error:
            failures.append(error)

    worker = threading.Thread(target=request, name="cancellable-json-request")

    # Act
    worker.start()
    assert response.read_started.wait(timeout=1.0)
    requester.cancel()
    worker.join(timeout=1.0)

    # Assert
    assert response.closed.is_set()
    assert worker.is_alive() is False
    assert len(failures) == 1
    assert isinstance(failures[0], CancelledError)
