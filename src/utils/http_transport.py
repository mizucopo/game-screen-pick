"""HTTP応答の受信期限と同一origin内のredirectを共有するtransport境界."""

from __future__ import annotations

import io
import socket
import time
from http.client import HTTPConnection, HTTPResponse, HTTPSConnection
from typing import Any, cast
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import (
    HTTPHandler,
    HTTPRedirectHandler,
    HTTPSHandler,
    Request,
    build_opener,
)


def _origin(url: str) -> tuple[str, str, int]:
    parts = urlsplit(url)
    if (
        parts.scheme not in {"http", "https"}
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
    ):
        raise ValueError("Invalid HTTP origin")
    default_port = 443 if parts.scheme == "https" else 80
    port = parts.port if parts.port is not None else default_port
    return parts.scheme, parts.hostname, port


class _SameOriginRedirectHandler(HTTPRedirectHandler):
    """接続先の変更とAuthorizationの別originへの転送を拒否する."""

    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> Request | None:
        try:
            same_origin = _origin(req.full_url) == _origin(newurl)
        except ValueError:
            same_origin = False
        if not same_origin:
            fp.close()
            raise URLError("Cross-origin HTTP redirects are not allowed")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class _DeadlineReader(io.RawIOBase):
    """status、header、chunk framing、bodyの各recvへ残り時間を適用する."""

    def __init__(self, connection: socket.socket, deadline: float) -> None:
        self._connection = connection
        self._raw = connection.makefile("rb", buffering=0)
        self._deadline = deadline

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int | None:
        try:
            remaining = self._deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("GPU runtime HTTP deadline exceeded")
            self._connection.settimeout(remaining)
            result = self._raw.readinto(buffer)
            if time.monotonic() >= self._deadline:
                raise TimeoutError("GPU runtime HTTP deadline exceeded")
            return result
        except BaseException:
            self._raw.close()
            raise

    def close(self) -> None:
        try:
            self._raw.close()
        finally:
            super().close()


def urlopen(request: Request, *, timeout: float) -> HTTPResponse:
    """lifecycle用HTTPを総受信deadline付きで開く。worker threadは作らない."""
    deadline = time.monotonic() + timeout

    class DeadlineResponse(HTTPResponse):
        def __init__(
            self,
            sock: socket.socket,
            debuglevel: int = 0,
            method: str | None = None,
            url: str | None = None,
        ) -> None:
            super().__init__(sock, debuglevel=debuglevel, method=method, url=url)
            original = self.fp
            self.fp = io.BufferedReader(_DeadlineReader(sock, deadline))
            if original is not None:
                original.close()

    class DeadlineHTTPConnection(HTTPConnection):
        response_class = DeadlineResponse

    class DeadlineHTTPSConnection(HTTPSConnection):
        response_class = DeadlineResponse

    class DeadlineHTTPHandler(HTTPHandler):
        def http_open(self, request: Request) -> HTTPResponse:
            return self.do_open(DeadlineHTTPConnection, request)

    class DeadlineHTTPSHandler(HTTPSHandler):
        def https_open(self, request: Request) -> HTTPResponse:
            return self.do_open(DeadlineHTTPSConnection, request)

    opener = build_opener(
        DeadlineHTTPHandler(), DeadlineHTTPSHandler(), _SameOriginRedirectHandler()
    )
    return cast(HTTPResponse, opener.open(request, timeout=timeout))
