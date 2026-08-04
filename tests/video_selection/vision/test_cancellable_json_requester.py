import threading
from concurrent.futures import CancelledError
from email.message import Message
from typing import Self
from urllib.error import HTTPError

import pytest

from src.video_selection.vision.cancellable_json_requester import (
    CancellableJsonRequester,
)


class BlockingConnection:
    """headerまたはbody待機中にcloseされるHTTP connection fake。"""

    def __init__(
        self,
        *,
        block_before_headers: bool,
        response_body: bytes | None = None,
    ) -> None:
        self._block_before_headers = block_before_headers
        self._response_body = response_body
        self.request_started = threading.Event()
        self.headers_started = threading.Event()
        self.read_started = threading.Event()
        self.closed = threading.Event()
        self.requests: list[tuple[str, str, bytes | None, dict[str, str]]] = []
        self.status = 200
        self.reason = "OK"
        self.headers = Message()
        self.sock = None

    def request(
        self,
        _method: str,
        _path: str,
        _body: bytes | None,
        _headers: dict[str, str],
    ) -> None:
        self.requests.append((_method, _path, _body, _headers))
        self.request_started.set()

    def getresponse(self) -> Self:
        self.headers_started.set()
        if self._block_before_headers:
            if not self.closed.wait(timeout=1.0):
                raise RuntimeError("HTTP connectionがcloseされませんでした")
            raise OSError("HTTP connection closed")
        return self

    def read(self, _size: int = -1) -> bytes:
        self.read_started.set()
        if self._response_body is not None:
            response_body = self._response_body
            self._response_body = b""
            return response_body
        if not self.closed.wait(timeout=1.0):
            raise RuntimeError("HTTP responseがcloseされませんでした")
        raise OSError("HTTP response closed")

    def close(self) -> None:
        self.closed.set()


@pytest.mark.parametrize("block_before_headers", (True, False))
def test_cancel_aborts_active_http_connection(
    monkeypatch: pytest.MonkeyPatch,
    block_before_headers: bool,
) -> None:
    """中止要求でheader・body待機中のHTTP connectionが終了されること。

    Arrange:
        - response headerまたはJSON body読込中で停止するconnectionが用意される
    Act:
        - 別threadのrequest中に中止が要求される
    Assert:
        - connectionが閉じられ、requestがCancelledErrorで終了すること
    """
    # Arrange
    connection = BlockingConnection(block_before_headers=block_before_headers)
    requester = CancellableJsonRequester()
    failures: list[BaseException] = []

    def open_connection(
        host: str,
        port: int | None,
        *,
        timeout: float,
    ) -> BlockingConnection:
        assert (host, port) == ("localhost", 11434)
        assert timeout == 60.0
        return connection

    monkeypatch.setattr(
        "src.video_selection.vision.cancellable_json_requester.HTTPConnection",
        open_connection,
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
    started = (
        connection.headers_started if block_before_headers else connection.read_started
    )
    assert started.wait(timeout=1.0)
    requester.cancel()
    worker.join(timeout=1.0)

    # Assert
    assert connection.closed.is_set()
    assert worker.is_alive() is False
    assert len(failures) == 1
    assert isinstance(failures[0], CancelledError)


def test_http_failure_retains_status_and_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTTP failureのstatusとRetry-Afterが呼出元へ維持されること。

    Arrange:
        - Retry-Afterを含むHTTP 429 responseが用意される
    Act:
        - JSON requestが実行される
    Assert:
        - HTTPErrorへstatusとRetry-Afterが維持されること
    """
    # Arrange
    connection = BlockingConnection(block_before_headers=False)
    connection.status = 429
    connection.reason = "rate limited"
    connection.headers["Retry-After"] = "30"
    requester = CancellableJsonRequester()

    def open_connection(
        _host: str,
        _port: int | None,
        *,
        timeout: float,
    ) -> BlockingConnection:
        assert timeout == 60.0
        return connection

    monkeypatch.setattr(
        "src.video_selection.vision.cancellable_json_requester.HTTPConnection",
        open_connection,
    )

    # Act
    # Assert
    with pytest.raises(HTTPError) as captured:
        requester("GET", "http://localhost:11434/api/tags", None, 60.0)
    assert captured.value.code == 429
    assert captured.value.headers["Retry-After"] == "30"


def test_valid_json_response_is_returned(monkeypatch: pytest.MonkeyPatch) -> None:
    """JSON responseとorigin-form requestが呼出元へ返されること。

    Arrange:
        - 正常なJSON bodyを返すHTTP connectionが用意される
    Act:
        - query付きURLへPOST requestが実行される
    Assert:
        - JSON値が返され、path、payload、headerが送信されること
    """
    # Arrange
    connection = BlockingConnection(
        block_before_headers=False,
        response_body=b'{"ready": true}',
    )
    requester = CancellableJsonRequester()

    def open_connection(
        _host: str,
        _port: int | None,
        *,
        timeout: float,
    ) -> BlockingConnection:
        assert timeout == 60.0
        return connection

    monkeypatch.setattr(
        "src.video_selection.vision.cancellable_json_requester.HTTPConnection",
        open_connection,
    )

    # Act
    result = requester(
        "POST",
        "http://localhost:11434/api/chat?stream=false",
        {"model": "vision"},
        60.0,
    )

    # Assert
    assert result == {"ready": True}
    assert connection.requests == [
        (
            "POST",
            "/api/chat?stream=false",
            b'{"model": "vision"}',
            {"Content-Type": "application/json"},
        )
    ]
