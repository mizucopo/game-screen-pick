"""中止可能なblocking JSON HTTP request境界。"""

import json
from collections.abc import Mapping
from concurrent.futures import CancelledError
from contextlib import suppress
from http.client import HTTPConnection, HTTPResponse, HTTPSConnection
from socket import SHUT_RDWR
from threading import Event, Lock, Thread
from urllib.error import HTTPError
from urllib.parse import SplitResult, urlsplit


class CancellableJsonRequester:
    """中止時にactive connectionとresponseを閉じるJSON requester。"""

    def __init__(self) -> None:
        self._cancellation = Event()
        self._active_work_lock = Lock()
        self._active_connections: set[HTTPConnection] = set()
        self._active_responses: set[HTTPResponse] = set()
        self._active_waiters: set[Event] = set()

    def __call__(
        self,
        method: str,
        url: str,
        payload: Mapping[str, object] | None,
        timeout: float,
    ) -> object:
        """中止可能なdaemon transportでJSON requestを実行する。"""
        self._require_active()
        completed = Event()
        values: list[object] = []
        failures: list[BaseException] = []

        def request() -> None:
            try:
                values.append(self._request_json(method, url, payload, timeout))
            except BaseException as error:
                failures.append(error)
            finally:
                completed.set()

        worker = Thread(target=request, name="cancellable-json-transport", daemon=True)
        with self._active_work_lock:
            self._require_active()
            self._active_waiters.add(completed)
            worker.start()
        try:
            completed.wait()
            self._require_active()
            if failures:
                raise failures[0]
            if len(values) != 1:
                raise AssertionError("JSON HTTP responseが確定していません")
            return values[0]
        finally:
            with self._active_work_lock:
                self._active_waiters.discard(completed)

    def cancel(self) -> None:
        """新規requestを拒否しactive transportを終了する。"""
        with self._active_work_lock:
            self._cancellation.set()
            waiters = tuple(self._active_waiters)
            responses = tuple(self._active_responses)
            connections = tuple(self._active_connections)
        for waiter in waiters:
            waiter.set()
        for response in responses:
            with suppress(Exception):
                response.close()
        for connection in connections:
            self._abort_connection(connection)

    def _request_json(
        self,
        method: str,
        url: str,
        payload: Mapping[str, object] | None,
        timeout: float,
    ) -> object:
        """connection確立からbody読込までを中止対象として実行する。"""
        target = urlsplit(url)
        connection = self._open_connection(target, timeout)
        response: HTTPResponse | None = None
        with self._active_work_lock:
            self._require_active()
            self._active_connections.add(connection)
        try:
            body = None if payload is None else json.dumps(payload).encode()
            connection.request(
                method,
                _request_path(target),
                body,
                {"Content-Type": "application/json"},
            )
            self._require_active()
            response = connection.getresponse()
            with self._active_work_lock:
                self._require_active()
                self._active_responses.add(response)
            if response.status >= 400:
                raise HTTPError(
                    url,
                    response.status,
                    str(response.reason),
                    response.headers,
                    response,
                )
            return json.load(response)
        except Exception:
            self._require_active()
            raise
        finally:
            with self._active_work_lock:
                self._active_connections.discard(connection)
                if response is not None:
                    self._active_responses.discard(response)
            if response is not None:
                with suppress(Exception):
                    response.close()
            with suppress(Exception):
                connection.close()

    @staticmethod
    def _open_connection(target: SplitResult, timeout: float) -> HTTPConnection:
        """URL schemeに対応するHTTP connectionを作成する。"""
        if target.scheme not in {"http", "https"} or target.hostname is None:
            raise ValueError("JSON HTTP request URLが不正です")
        connection_type = (
            HTTPSConnection if target.scheme == "https" else HTTPConnection
        )
        return connection_type(target.hostname, target.port, timeout=timeout)

    @staticmethod
    def _abort_connection(connection: HTTPConnection) -> None:
        """header待機も解除するためsocketをshutdownしてconnectionを閉じる。"""
        if connection.sock is not None:
            with suppress(OSError):
                connection.sock.shutdown(SHUT_RDWR)
        with suppress(Exception):
            connection.close()

    def _require_active(self) -> None:
        """中止要求後のrequest処理を拒否する。"""
        if self._cancellation.is_set():
            raise CancelledError("JSON HTTP requestは中止されました")


def _request_path(target: SplitResult) -> str:
    """HTTP origin-formのpathとqueryを返す。"""
    path = target.path or "/"
    return f"{path}?{target.query}" if target.query else path
