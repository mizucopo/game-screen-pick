"""利用者が明示したvLLM起動・停止とOllama解放を実行する."""

from __future__ import annotations

import errno
import io
import json
import logging
import math
import re
import shlex
import socket
import subprocess
import time
from http.client import HTTPConnection, HTTPResponse, HTTPSConnection
from types import TracebackType
from typing import Any, cast
from urllib.error import HTTPError, URLError
from urllib.request import HTTPHandler, HTTPSHandler, Request, build_opener

from ..models.vllm_config import VllmConfig
from ..models.vllm_runtime_config import VllmRuntimeConfig

logger = logging.getLogger(__name__)


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

    opener = build_opener(DeadlineHTTPHandler(), DeadlineHTTPSHandler())
    return cast(HTTPResponse, opener.open(request, timeout=timeout))


class VllmRuntimeSession:
    """最初の推論要求まで待ち、試行した起動だけを終了時に片付ける."""

    def __init__(
        self,
        config: VllmRuntimeConfig,
        vllm_config: VllmConfig,
        *,
        ollama_host: str | None = None,
        ollama_api_key: str | None = None,
        ollama_timeout_seconds: float = 900,
    ) -> None:
        self.config = config
        self.vllm_config = vllm_config
        self.ollama_host = ollama_host
        self.ollama_api_key = ollama_api_key
        self.ollama_timeout_seconds = ollama_timeout_seconds
        self._ready = False
        self._start_attempted = False
        self._closed = False
        self._acquisition_error: BaseException | None = None
        self._health_url = vllm_config.base_url.removesuffix("/v1") + "/health"
        self._models_url = vllm_config.base_url + "/models"

    def __enter__(self) -> VllmRuntimeSession:
        return self

    def ensure_ready(self) -> None:
        """推論要求が発生したときだけ準備する."""
        if self._closed:
            raise RuntimeError("終了したvLLM runtime sessionは再利用できません")
        if self._acquisition_error is not None:
            raise self._acquisition_error
        if self._ready:
            return
        try:
            self.check_ollama_available()
            if self.config.unload_ollama:
                self._unload_ollama()
            if self.config.start_command:
                self._start_attempted = True
                self._run_command(self.config.start_command, "起動")
                deadline = time.monotonic() + self.config.startup_timeout_seconds
                logger.info("vLLMの応答を待機しています")
                while not self._is_ready(deadline):
                    self._pause(deadline, "vLLM起動確認")
                logger.info("vLLMの起動を確認しました")
            self._ready = True
        except BaseException as error:
            self._acquisition_error = error
            raise

    def check_ollama_available(self) -> None:
        """管理対象vLLMが既に応答中ならOllama推論や起動を始めない."""
        if not self.config.start_command:
            return
        deadline = time.monotonic() + self.config.startup_timeout_seconds
        if self._server_listening(deadline):
            raise RuntimeError("vLLM endpointが既に応答中のためruntimeを開始できません")

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, traceback
        if self._closed:
            return
        self._closed = True
        if not self._start_attempted:
            return
        try:
            self._run_command(self.config.stop_command, "停止")
            deadline = time.monotonic() + self.config.shutdown_timeout_seconds
            logger.info("vLLM endpointの停止を待機しています")
            while time.monotonic() < deadline:
                if not self._server_listening(deadline):
                    logger.info("vLLMの停止を確認しました")
                    return
                self._pause(deadline, "vLLM停止確認")
            raise RuntimeError("vLLM停止確認が時間切れになりました")
        except BaseException:
            if exc is None:
                raise
            logger.error("vLLM停止処理にも失敗しました。元の処理エラーを維持します")

    def _run_command(self, command: tuple[str, ...], stage: str) -> None:
        logger.info("vLLM%scommandを実行します", stage)
        arguments = [
            re.sub(
                r"\{model(_shell)?\}",
                lambda match: (
                    shlex.quote(self.vllm_config.model)
                    if match[1]
                    else self.vllm_config.model
                ),
                part,
            )
            for part in command
        ]
        try:
            subprocess.run(
                arguments,
                shell=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
                timeout=self.config.command_timeout_seconds,
            )
        except (OSError, subprocess.SubprocessError):
            raise RuntimeError(
                f"vLLM{stage}commandが失敗または時間切れになりました"
            ) from None

    def _unload_ollama(self) -> None:
        if not self.ollama_host:
            raise ValueError("Ollama解放には接続先が必要です")
        if (
            not math.isfinite(self.ollama_timeout_seconds)
            or self.ollama_timeout_seconds <= 0
        ):
            raise ValueError("Ollama解放timeoutは有限の正の数が必要です")
        deadline = time.monotonic() + self.ollama_timeout_seconds
        host = self.ollama_host.rstrip("/")
        logger.info("Ollamaのloaded modelを確認しています")
        models = self._ollama_models(host, deadline)
        logger.info("Ollamaのモデル解放を開始します: %d件", len(models))
        for model in models:
            self._request(
                host + "/api/generate",
                self.ollama_api_key,
                deadline,
                payload={"model": model, "keep_alive": 0, "stream": False},
            )
        while models:
            models = self._ollama_models(host, deadline)
            if models:
                self._pause(deadline, "Ollama解放確認")
        logger.info("Ollamaのモデル解放を確認しました")

    def _ollama_models(self, host: str, deadline: float) -> list[str]:
        response = self._json_response(
            self._request(host + "/api/ps", self.ollama_api_key, deadline)
        )
        models = response.get("models")
        if not isinstance(models, list):
            raise ValueError("Ollama /api/psの応答にmodels配列がありません")
        names = []
        for model in models:
            if not isinstance(model, dict):
                raise ValueError("Ollama loaded modelの応答が不正です")
            name = model.get("name") or model.get("model")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Ollama loaded modelの名前が不正です")
            names.append(name)
        return sorted(set(names))

    def _is_ready(self, deadline: float) -> bool:
        try:
            self._request(self._health_url, self.vllm_config.api_key, deadline)
            response = self._json_response(
                self._request(self._models_url, self.vllm_config.api_key, deadline)
            )
        except (RuntimeError, ValueError):
            return False
        models = response.get("data")
        return isinstance(models, list) and any(
            isinstance(model, dict) and model.get("id") == self.vllm_config.model
            for model in models
        )

    def _server_listening(self, deadline: float) -> bool:
        return any(
            self._endpoint_listening(url, deadline)
            for url in (self._health_url, self._models_url)
        )

    def _endpoint_listening(self, url: str, deadline: float) -> bool:
        request = self._make_request(url, self.vllm_config.api_key)
        try:
            with urlopen(request, timeout=self._request_timeout(deadline)):
                self._request_timeout(deadline)
                return True
        except HTTPError as error:
            error.close()
            return True
        except (URLError, OSError) as error:
            reason = error.reason if isinstance(error, URLError) else error
            if isinstance(reason, ConnectionRefusedError) or (
                isinstance(reason, OSError) and reason.errno == errno.ECONNREFUSED
            ):
                return False
            if isinstance(reason, ConnectionResetError) or (
                isinstance(reason, OSError) and reason.errno == errno.ECONNRESET
            ):
                return True
            raise RuntimeError("vLLM endpointの接続状態を確認できません") from None

    def _request(
        self,
        url: str,
        api_key: str | None,
        deadline: float,
        *,
        payload: dict[str, Any] | None = None,
    ) -> bytes:
        request = self._make_request(url, api_key, payload=payload)
        try:
            with urlopen(request, timeout=self._request_timeout(deadline)) as response:
                body = bytes(response.read())
                self._request_timeout(deadline)
                return body
        except HTTPError as error:
            error.close()
            raise RuntimeError("GPU runtime HTTP要求に失敗しました") from None
        except (URLError, OSError):
            raise RuntimeError("GPU runtime HTTP要求に失敗しました") from None

    @staticmethod
    def _make_request(
        url: str,
        api_key: str | None,
        *,
        payload: dict[str, Any] | None = None,
    ) -> Request:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return Request(
            url,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers=headers,
            method="POST" if payload is not None else "GET",
        )

    @staticmethod
    def _json_response(body: bytes) -> dict[str, Any]:
        try:
            response: Any = json.loads(body)
        except (ValueError, UnicodeError):
            raise ValueError("GPU runtime HTTP応答が有効なJSONではありません") from None
        if not isinstance(response, dict):
            raise ValueError("GPU runtime HTTP応答はJSON objectが必要です")
        return response

    @staticmethod
    def _request_timeout(deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("GPU runtime確認が時間切れになりました")
        return min(30.0, remaining)

    @staticmethod
    def _pause(deadline: float, stage: str) -> None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(f"{stage}が時間切れになりました")
        time.sleep(min(0.5, remaining))
