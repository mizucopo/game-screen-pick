"""共有HTTP transportのredirect契約を実HTTPで検証する."""

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.error import URLError
from urllib.request import Request

import pytest

from src.utils.http_transport import urlopen


class QuietHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        del format, args


@contextmanager
def serve_http(handler: type[BaseHTTPRequestHandler]) -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    worker = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}
    )
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=1)
        assert not worker.is_alive()


@pytest.mark.parametrize("change", ["port", "host", "scheme"])
def test_cross_origin_redirect_never_contacts_target_or_forwards_key(
    change: str,
) -> None:
    target_keys: list[str | None] = []

    class Target(QuietHandler):
        def do_GET(self) -> None:
            target_keys.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.end_headers()

    with serve_http(Target) as target:
        if change == "host":
            target = target.replace("127.0.0.1", "localhost")
        elif change == "scheme":
            target = target.replace("http://", "https://")

        class Redirect(QuietHandler):
            def do_GET(self) -> None:
                self.send_response(302)
                self.send_header("Location", target + "/models")
                self.end_headers()

        with serve_http(Redirect) as origin:
            request = Request(
                origin + "/models", headers={"Authorization": "Bearer secret"}
            )
            with pytest.raises(URLError, match="origin"):
                urlopen(request, timeout=1)
    assert target_keys == []


@pytest.mark.parametrize("absolute", [False, True])
def test_same_origin_redirect_preserves_authorization(absolute: bool) -> None:
    seen: list[tuple[str, str | None]] = []

    class Handler(QuietHandler):
        def do_GET(self) -> None:
            seen.append((self.path, self.headers.get("Authorization")))
            if self.path == "/models":
                self.send_response(302)
                location = "/final"
                if absolute:
                    location = origin + location
                self.send_header("Location", location)
                self.end_headers()
            else:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"data":[]}')

    with serve_http(Handler) as origin:
        request = Request(
            origin + "/models", headers={"Authorization": "Bearer secret"}
        )
        with urlopen(request, timeout=1) as response:
            assert response.read() == b'{"data":[]}'
    assert seen == [("/models", "Bearer secret"), ("/final", "Bearer secret")]
