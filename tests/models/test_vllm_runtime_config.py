"""任意のvLLM起動停止設定を検証する."""

from dataclasses import replace
from typing import Any

import pytest

from src.models.vllm_runtime_config import VllmRuntimeConfig


def test_runtime_defaults_do_not_enable_lifecycle_actions() -> None:
    config = VllmRuntimeConfig()

    assert config.unload_ollama is False
    assert config.start_command == ()
    assert config.stop_command == ()
    assert config.command_timeout_seconds == 60
    assert config.startup_timeout_seconds == 900
    assert config.shutdown_timeout_seconds == 60
    assert config.enabled is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("unload_ollama", "true"),
        ("unload_ollama", 1),
        ("start_command", ("start",)),
        ("stop_command", ("stop",)),
        ("start_command", "start"),
        ("stop_command", ["stop"]),
        ("command_timeout_seconds", 0),
        ("startup_timeout_seconds", float("inf")),
        ("shutdown_timeout_seconds", float("nan")),
        ("command_timeout_seconds", True),
        ("startup_timeout_seconds", "60"),
        ("shutdown_timeout_seconds", -1),
    ],
)
def test_invalid_runtime_settings_are_rejected(field: str, value: Any) -> None:
    with pytest.raises(ValueError):
        replace(VllmRuntimeConfig(), **{field: value})


@pytest.mark.parametrize("argument", ["", "  ", "bad\x00argument", 1])
def test_commands_require_nonempty_string_arguments(argument: Any) -> None:
    with pytest.raises(ValueError, match="command"):
        VllmRuntimeConfig(start_command=("runner", argument), stop_command=("stop",))


def test_only_explicit_actions_enable_runtime_management() -> None:
    assert VllmRuntimeConfig(unload_ollama=True).enabled
    assert VllmRuntimeConfig(
        start_command=("runner", "{model}"), stop_command=("stop",)
    ).enabled
    assert not VllmRuntimeConfig(startup_timeout_seconds=30).enabled
