"""任意のGPU lifecycleを公開session境界で検証する."""

import errno
import io
import json
import logging
import os
import shlex
import subprocess
import sys
from email.message import Message
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request

import pytest

from src.models.vllm_config import VllmConfig
from src.models.vllm_runtime_config import VllmRuntimeConfig
from src.services.vllm_runtime_session import VllmRuntimeSession


class RuntimeTransport:
    """外部HTTPとprocessの状態を公開通信契約で模倣する."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.history: list[str] = []
        self.requests: list[Request] = []
        self.commands: list[list[str]] = []
        self.live = False
        self.loaded_models = ["ollama-model"]
        self.served_model = "game-model"
        self.start_failure: BaseException | None = None
        self.stop_failure: BaseException | None = None
        self.unload_completes = True
        self.stop_completes = True
        self.clock = 0.0
        monkeypatch.setattr("src.services.vllm_runtime_session.urlopen", self.http)
        monkeypatch.setattr(
            "src.services.vllm_runtime_session.subprocess.run", self.run
        )
        monkeypatch.setattr(
            "src.services.vllm_runtime_session.time.monotonic", self.now
        )
        monkeypatch.setattr("src.services.vllm_runtime_session.time.sleep", self.sleep)

    def now(self) -> float:
        return self.clock

    def sleep(self, seconds: float) -> None:
        self.clock += seconds

    def http(self, request: Request, *, timeout: float) -> io.BytesIO:
        assert 0 < timeout <= 900
        self.requests.append(request)
        self.history.append(request.full_url)
        if request.full_url.endswith("/api/ps"):
            return io.BytesIO(
                json.dumps(
                    {"models": [{"name": name} for name in self.loaded_models]}
                ).encode()
            )
        if request.full_url.endswith("/api/generate"):
            assert isinstance(request.data, bytes)
            payload = json.loads(request.data)
            assert payload == {
                "model": "ollama-model",
                "keep_alive": 0,
                "stream": False,
            }
            if self.unload_completes:
                self.loaded_models.clear()
            return io.BytesIO(b'{"done":true}')
        if not self.live:
            raise URLError(ConnectionRefusedError(errno.ECONNREFUSED, "offline"))
        if request.full_url.endswith("/health"):
            return io.BytesIO(b"")
        assert request.full_url.endswith("/v1/models")
        return io.BytesIO(json.dumps({"data": [{"id": self.served_model}]}).encode())

    def run(
        self, command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[bytes]:
        assert kwargs["shell"] is False
        assert kwargs["timeout"] == 60
        assert kwargs["stdin"] == subprocess.DEVNULL
        assert kwargs["stdout"] == subprocess.DEVNULL
        assert kwargs["stderr"] == subprocess.DEVNULL
        assert "capture_output" not in kwargs
        self.commands.append(command)
        self.history.append(command[0])
        failure = self.start_failure if command[0] == "start" else self.stop_failure
        if failure is not None:
            raise failure
        if command[0] == "stop" and not self.stop_completes:
            return subprocess.CompletedProcess(command, 0, b"", b"")
        self.live = command[0] == "start"
        return subprocess.CompletedProcess(command, 0, b"", b"")


def test_disabled_or_unused_session_does_not_call_process_or_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_http(request: Request, **kwargs: Any) -> None:
        del request, kwargs
        pytest.fail("passive lifecycle must not contact a server")

    def no_process(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        del args, kwargs
        pytest.fail("passive lifecycle must not run commands")

    monkeypatch.setattr("src.services.vllm_runtime_session.urlopen", no_http)
    monkeypatch.setattr("src.services.vllm_runtime_session.subprocess.run", no_process)
    model = VllmConfig(model="game-model")

    with VllmRuntimeSession(VllmRuntimeConfig(), model) as session:
        session.ensure_ready()
        session.ensure_ready()
    with VllmRuntimeSession(
        VllmRuntimeConfig(start_command=("start",), stop_command=("stop",)), model
    ):
        pass


def test_lifecycle_releases_ollama_then_lazily_starts_and_stops_owned_vllm(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    transport = RuntimeTransport(monkeypatch)
    config = VllmRuntimeConfig(
        unload_ollama=True,
        start_command=("start", "{model}", "literal $() ;"),
        stop_command=("stop", "{model}"),
    )
    model = VllmConfig(
        model="game-model", base_url="http://vllm/proxy/v1", api_key="vllm-key"
    )

    with VllmRuntimeSession(
        config, model, ollama_host="http://ollama:11434", ollama_api_key="ollama-key"
    ) as session:
        assert transport.history == []
        session.ensure_ready()
        prepared = list(transport.history)
        session.ensure_ready()
        assert transport.history == prepared
        assert transport.live

    assert transport.commands == [
        ["start", "game-model", "literal $() ;"],
        ["stop", "game-model"],
    ]
    assert transport.history.index(
        "http://ollama:11434/api/generate"
    ) < transport.history.index("start")
    assert transport.history.index("start") < transport.history.index("stop")
    assert not transport.live
    assert "http://vllm/proxy/health" in transport.history
    assert "http://vllm/proxy/v1/models" in transport.history
    for request in transport.requests:
        key = "ollama-key" if "/api/" in request.full_url else "vllm-key"
        assert request.get_header("Authorization") == f"Bearer {key}"
    assert "Ollamaのモデル解放を開始します: 1件" in caplog.text
    assert "Ollamaのモデル解放を確認しました" in caplog.text
    assert "vLLMの起動を確認しました" in caplog.text
    assert "vLLMの停止を確認しました" in caplog.text
    assert "literal $() ;" not in caplog.text
    assert "vllm-key" not in caplog.text
    assert "ollama-key" not in caplog.text


def test_unload_only_does_not_probe_start_or_stop_vllm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = RuntimeTransport(monkeypatch)

    with VllmRuntimeSession(
        VllmRuntimeConfig(unload_ollama=True),
        VllmConfig(model="game-model"),
        ollama_host="http://ollama:11434",
    ) as session:
        session.ensure_ready()

    assert transport.commands == []
    assert transport.loaded_models == []
    assert all("http://ollama:11434/api/" in url for url in transport.history)


def managed_config(**overrides: Any) -> VllmRuntimeConfig:
    values = {
        "start_command": ("start",),
        "stop_command": ("stop",),
        "startup_timeout_seconds": 1.0,
        "shutdown_timeout_seconds": 1.0,
    }
    return VllmRuntimeConfig(**{**values, **overrides})


def test_existing_server_prevents_unload_start_and_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = RuntimeTransport(monkeypatch)
    transport.live = True

    with (
        pytest.raises(RuntimeError, match="既に応答"),
        VllmRuntimeSession(
            managed_config(unload_ollama=True),
            VllmConfig(model="game-model"),
            ollama_host="http://ollama:11434",
        ) as session,
    ):
        session.ensure_ready()

    assert transport.commands == []
    assert transport.loaded_models == ["ollama-model"]
    assert not any("/api/" in url for url in transport.history)


def test_local_ollama_preflight_is_read_only_and_ignores_unmanaged_vllm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = RuntimeTransport(monkeypatch)
    transport.live = True
    with VllmRuntimeSession(
        VllmRuntimeConfig(unload_ollama=True), VllmConfig(model="game-model")
    ) as passive:
        passive.check_ollama_available()
    assert transport.history == []

    with (
        VllmRuntimeSession(managed_config(), VllmConfig(model="game-model")) as managed,
        pytest.raises(RuntimeError, match="既に応答"),
    ):
        managed.check_ollama_available()

    assert transport.commands == []
    assert transport.loaded_models == ["ollama-model"]


def test_http_error_still_counts_as_existing_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = RuntimeTransport(monkeypatch)

    def unavailable_http(request: Request, *, timeout: float) -> io.BytesIO:
        del timeout
        raise HTTPError(request.full_url, 503, "secret-body", Message(), None)

    monkeypatch.setattr("src.services.vllm_runtime_session.urlopen", unavailable_http)
    with (
        pytest.raises(RuntimeError, match="既に応答"),
        VllmRuntimeSession(managed_config(), VllmConfig(model="game-model")) as session,
    ):
        session.ensure_ready()
    assert transport.commands == []


@pytest.mark.parametrize(
    "failure",
    [
        subprocess.CalledProcessError(1, ["start"], output=b"private-key"),
        subprocess.TimeoutExpired(["start"], 60, output=b"private-key"),
        KeyboardInterrupt(),
    ],
)
def test_attempted_start_is_stopped_even_when_command_fails_or_is_interrupted(
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
) -> None:
    transport = RuntimeTransport(monkeypatch)
    transport.start_failure = failure
    expected_error = (
        KeyboardInterrupt if isinstance(failure, KeyboardInterrupt) else RuntimeError
    )

    with (
        pytest.raises(expected_error) as error,
        VllmRuntimeSession(managed_config(), VllmConfig(model="game-model")) as session,
    ):
        session.ensure_ready()

    assert transport.commands == [["start"], ["stop"]]
    assert "private-key" not in str(error.value)


def test_wrong_served_model_reaches_deadline_and_stops_started_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = RuntimeTransport(monkeypatch)
    transport.served_model = "other-model"

    with (
        pytest.raises(RuntimeError, match="起動確認.*時間切れ"),
        VllmRuntimeSession(managed_config(), VllmConfig(model="game-model")) as session,
    ):
        session.ensure_ready()

    assert transport.clock == 1.0
    assert transport.commands == [["start"], ["stop"]]


def test_unload_timeout_prevents_start_and_unowned_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = RuntimeTransport(monkeypatch)
    transport.unload_completes = False

    with (
        pytest.raises(RuntimeError, match="時間切れ"),
        VllmRuntimeSession(
            managed_config(unload_ollama=True),
            VllmConfig(model="game-model"),
            ollama_host="http://ollama:11434",
            ollama_timeout_seconds=1.0,
        ) as session,
    ):
        session.ensure_ready()

    assert transport.clock == 1.0
    assert transport.commands == []


@pytest.mark.parametrize(
    "original", [ValueError("original failure"), KeyboardInterrupt()]
)
def test_cleanup_failure_preserves_original_processing_exception(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    original: BaseException,
) -> None:
    transport = RuntimeTransport(monkeypatch)
    transport.stop_failure = subprocess.CalledProcessError(
        1, ["stop"], output=b"private-key"
    )

    with (
        caplog.at_level(logging.ERROR),
        pytest.raises(type(original)) as error,
        VllmRuntimeSession(managed_config(), VllmConfig(model="game-model")) as session,
    ):
        session.ensure_ready()
        raise original

    assert error.value is original
    assert transport.commands == [["start"], ["stop"]]
    assert "停止処理" in caplog.text
    assert "private-key" not in caplog.text


@pytest.mark.parametrize("command_fails", [True, False])
def test_successful_processing_reports_failed_or_incomplete_stop(
    monkeypatch: pytest.MonkeyPatch,
    command_fails: bool,
) -> None:
    transport = RuntimeTransport(monkeypatch)
    if command_fails:
        transport.stop_failure = subprocess.CalledProcessError(1, ["stop"])
    else:
        transport.stop_completes = False

    with (
        pytest.raises(RuntimeError, match="停止"),
        VllmRuntimeSession(managed_config(), VllmConfig(model="game-model")) as session,
    ):
        session.ensure_ready()

    assert transport.commands == [["start"], ["stop"]]


def test_real_subprocess_hooks_receive_literal_model_and_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "runtime.json"
    model = "game-model; echo never-a-shell"
    commands = (
        (
            sys.executable,
            "-c",
            "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text(sys.argv[2])",
            str(marker),
            "{model}",
        ),
        (
            sys.executable,
            "-c",
            "import pathlib,sys; pathlib.Path(sys.argv[1]).unlink()",
            str(marker),
        ),
    )

    def http(request: Request, *, timeout: float) -> io.BytesIO:
        assert timeout > 0
        if not marker.exists():
            raise URLError(ConnectionRefusedError(errno.ECONNREFUSED, "offline"))
        if request.full_url.endswith("/health"):
            return io.BytesIO(b"")
        return io.BytesIO(json.dumps({"data": [{"id": marker.read_text()}]}).encode())

    monkeypatch.setattr("src.services.vllm_runtime_session.urlopen", http)
    with VllmRuntimeSession(
        VllmRuntimeConfig(start_command=commands[0], stop_command=commands[1]),
        VllmConfig(model=model),
    ) as session:
        session.ensure_ready()
        assert marker.read_text() == model
    assert not marker.exists()


def test_model_shell_placeholder_is_quoted_once_while_model_stays_literal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = RuntimeTransport(monkeypatch)
    model = "game model's $HOME; {model}"
    quoted = """'game model'"'"'s $HOME; {model}'"""
    transport.served_model = model
    with VllmRuntimeSession(
        managed_config(
            start_command=("start", "{model}", "--remote={model_shell}"),
            stop_command=("stop", "{model_shell}"),
        ),
        VllmConfig(model=model),
    ) as session:
        session.ensure_ready()

    assert transport.commands == [
        ["start", model, "--remote=" + quoted],
        ["stop", quoted],
    ]


def test_model_shell_survives_remote_shell_as_one_literal_argument(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "runtime.json"
    received = tmp_path / "received.json"
    unexpected = tmp_path / "must-not-be-created"
    model = f"game 'model' $HOME; touch {unexpected}; #"
    remote = tmp_path / "ssh_standin.py"
    remote.write_text(
        "import subprocess, sys\n"
        "subprocess.run(['/bin/sh', '-c', ' '.join(sys.argv[1:])], check=True)\n",
        encoding="utf-8",
    )
    control = tmp_path / "control.py"
    control.write_text(
        "import json, pathlib, sys\n"
        "marker = pathlib.Path(sys.argv[2])\n"
        "if sys.argv[1] == 'start':\n"
        "    marker.write_text(json.dumps(sys.argv[4:]))\n"
        "    pathlib.Path(sys.argv[3]).write_text(marker.read_text())\n"
        "else:\n"
        "    marker.unlink()\n",
        encoding="utf-8",
    )
    prefix = (
        sys.executable,
        str(remote),
        shlex.quote(sys.executable),
        shlex.quote(str(control)),
    )

    def http(request: Request, *, timeout: float) -> io.BytesIO:
        assert timeout > 0
        if not marker.exists():
            raise URLError(ConnectionRefusedError(errno.ECONNREFUSED, "offline"))
        if request.full_url.endswith("/health"):
            return io.BytesIO(b"")
        served = json.loads(marker.read_text())[0]
        return io.BytesIO(json.dumps({"data": [{"id": served}]}).encode())

    monkeypatch.setattr("src.services.vllm_runtime_session.urlopen", http)
    with VllmRuntimeSession(
        VllmRuntimeConfig(
            start_command=(
                *prefix,
                "start",
                shlex.quote(str(marker)),
                shlex.quote(str(received)),
                "{model_shell}",
            ),
            stop_command=(*prefix, "stop", shlex.quote(str(marker))),
            startup_timeout_seconds=0.1,
        ),
        VllmConfig(model=model),
    ) as session:
        session.ensure_ready()
        assert json.loads(received.read_text()) == [model]
        assert not unexpected.exists()
    assert not marker.exists()


def test_detached_child_inheriting_streams_does_not_block_control_hook(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "child.pid"
    control = tmp_path / "control.py"
    control.write_text(
        "import os, pathlib, signal, subprocess, sys\n"
        "marker = pathlib.Path(sys.argv[2])\n"
        "if sys.argv[1] == 'start':\n"
        "    child_code = 'import time; time.sleep(2)'\n"
        "    child = subprocess.Popen([sys.executable, '-c', child_code])\n"
        "    marker.write_text(str(child.pid))\n"
        "else:\n"
        "    try:\n"
        "        os.kill(int(marker.read_text()), signal.SIGTERM)\n"
        "    except ProcessLookupError:\n"
        "        pass\n"
        "    marker.unlink()\n",
        encoding="utf-8",
    )

    def http(request: Request, *, timeout: float) -> io.BytesIO:
        assert timeout > 0
        if not marker.exists():
            raise URLError(ConnectionRefusedError(errno.ECONNREFUSED, "offline"))
        if request.full_url.endswith("/health"):
            return io.BytesIO(b"")
        return io.BytesIO(b'{"data":[{"id":"game-model"}]}')

    monkeypatch.setattr("src.services.vllm_runtime_session.urlopen", http)
    with VllmRuntimeSession(
        VllmRuntimeConfig(
            start_command=(sys.executable, str(control), "start", str(marker)),
            stop_command=(sys.executable, str(control), "stop", str(marker)),
            command_timeout_seconds=0.5,
        ),
        VllmConfig(model="game-model"),
    ) as session:
        session.ensure_ready()
        os.kill(int(marker.read_text()), 0)
    assert not marker.exists()


def test_repeated_acquisition_preserves_original_error_without_restarting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = RuntimeTransport(monkeypatch)
    transport.start_failure = subprocess.CalledProcessError(1, ["start"])
    with VllmRuntimeSession(
        managed_config(), VllmConfig(model="game-model")
    ) as session:
        with pytest.raises(RuntimeError, match="起動command") as first:
            session.ensure_ready()
        with pytest.raises(RuntimeError, match="起動command") as retry:
            session.ensure_ready()
        assert retry.value is first.value
        assert transport.commands == [["start"]]
    assert transport.commands == [["start"], ["stop"]]
    with pytest.raises(RuntimeError, match="終了"):
        session.ensure_ready()
