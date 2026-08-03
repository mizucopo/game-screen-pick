"""中止可能なblocking JSON HTTP request境界。"""

import json
from collections.abc import Mapping
from concurrent.futures import CancelledError
from contextlib import suppress
from http.client import HTTPResponse
from threading import Event, Lock
from urllib.request import Request, urlopen


class CancellableJsonRequester:
    """中止時にactive responseを閉じるJSON requester。"""

    def __init__(self) -> None:
        self._cancellation = Event()
        self._active_response_lock = Lock()
        self._active_responses: set[HTTPResponse] = set()

    def __call__(
        self,
        method: str,
        url: str,
        payload: Mapping[str, object] | None,
        timeout: float,
    ) -> object:
        """JSON requestを実行し中止されたblocking readを終了する。"""
        self._require_active()
        body = None if payload is None else json.dumps(payload).encode()
        request = Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        with urlopen(request, timeout=timeout) as response:
            with self._active_response_lock:
                if self._cancellation.is_set():
                    response.close()
                    raise CancelledError("JSON HTTP requestは中止されました")
                self._active_responses.add(response)
            try:
                return json.load(response)
            except Exception:
                self._require_active()
                raise
            finally:
                with self._active_response_lock:
                    self._active_responses.discard(response)

    def cancel(self) -> None:
        """新規requestを拒否しactive responseを閉じる。"""
        with self._active_response_lock:
            self._cancellation.set()
            responses = tuple(self._active_responses)
        for response in responses:
            with suppress(Exception):
                response.close()

    def _require_active(self) -> None:
        """中止要求後のrequest処理を拒否する。"""
        if self._cancellation.is_set():
            raise CancelledError("JSON HTTP requestは中止されました")
