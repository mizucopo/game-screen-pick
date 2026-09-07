"""VideoRunConfigLoaderの単体テスト."""

from pathlib import Path

import pytest

from src.utils.video_run_config_loader import VideoRunConfigLoader


def test_load_returns_only_explicit_run_values(tmp_path: Path) -> None:
    """明示したrun設定を公開型へ正規化して返すこと."""
    config_path = tmp_path / "picker.toml"
    config_path.write_text(
        """[run]
primary_model = "primary"
ollama_api_key = "ollama-secret"
openai_api_key = "openai-secret"
gemini_api_key = "gemini-secret"
xai_api_key = "xai-secret"
ollama_timeout = 120
allow_cpu = true
sample_interval_seconds = 2.5
""",
        encoding="utf-8",
    )

    assert VideoRunConfigLoader.load(str(config_path)) == {
        "primary_model": "primary",
        "ollama_api_key": "ollama-secret",
        "openai_api_key": "openai-secret",
        "gemini_api_key": "gemini-secret",
        "xai_api_key": "xai-secret",
        "ollama_timeout": 120.0,
        "allow_cpu": True,
        "sample_interval_seconds": 2.5,
    }


def test_runtime_settings_keep_command_arguments_and_explicit_types(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "runtime.toml"
    config_path.write_text(
        "[run]\nollama_unload_before_vllm = true\n"
        'vllm_start_command = ["runner", "start", "{model}", "path with spaces"]\n'
        'vllm_stop_command = ["runner", "stop"]\n'
        "vllm_command_timeout = 12\n"
        "vllm_startup_timeout = 45.5\n"
        "vllm_shutdown_timeout = 20\n",
        encoding="utf-8",
    )

    assert VideoRunConfigLoader.load(str(config_path)) == {
        "ollama_unload_before_vllm": True,
        "vllm_start_command": ("runner", "start", "{model}", "path with spaces"),
        "vllm_stop_command": ("runner", "stop"),
        "vllm_command_timeout": 12.0,
        "vllm_startup_timeout": 45.5,
        "vllm_shutdown_timeout": 20.0,
    }


@pytest.mark.parametrize(
    "setting",
    [
        'ollama_unload_before_vllm = "true"',
        'vllm_start_command = "runner start"',
        'vllm_stop_command = ["runner", 42]',
        'vllm_start_command = ["runner", ""]',
        "vllm_command_timeout = true",
    ],
)
def test_runtime_config_rejects_wrong_public_types(
    tmp_path: Path, setting: str
) -> None:
    config_path = tmp_path / "runtime.toml"
    config_path.write_text(f"[run]\n{setting}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=setting.split(" = ")[0]):
        VideoRunConfigLoader.load(str(config_path))
