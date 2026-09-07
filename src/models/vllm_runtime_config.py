"""推論条件と分離した、任意のvLLM起動停止設定."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class VllmRuntimeConfig:
    """利用者が明示したOllama解放とvLLM起動停止だけを有効にする."""

    unload_ollama: bool = False
    start_command: tuple[str, ...] = ()
    stop_command: tuple[str, ...] = ()
    command_timeout_seconds: float = 60.0
    startup_timeout_seconds: float = 900.0
    shutdown_timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        """副作用を始める前にargvとtimeoutの契約を確認する."""
        if not isinstance(self.unload_ollama, bool):
            raise ValueError("ollama_unload_before_vllmはbooleanで指定してください")
        for name in ("start_command", "stop_command"):
            command = getattr(self, name)
            if not isinstance(command, tuple) or any(
                not isinstance(argument, str)
                or not argument.strip()
                or "\x00" in argument
                for argument in command
            ):
                raise ValueError(f"vllm_{name}は空でない文字列のtupleが必要です")
        if bool(self.start_command) != bool(self.stop_command):
            raise ValueError(
                "vllm_start_commandとvllm_stop_commandは両方指定してください"
            )
        for name in (
            "command_timeout_seconds",
            "startup_timeout_seconds",
            "shutdown_timeout_seconds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"vllm_{name}は有限の正の数が必要です")
            try:
                seconds = float(value)
            except OverflowError as error:
                raise ValueError(f"vllm_{name}は有限の正の数が必要です") from error
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError(f"vllm_{name}は有限の正の数が必要です")

    @property
    def enabled(self) -> bool:
        """起動停止またはOllama解放が明示されているかを返す."""
        return self.unload_ollama or bool(self.start_command)
