"""Ollama接続設定."""

from dataclasses import dataclass


@dataclass(frozen=True)
class OllamaConfig:
    """Ollama scene分類に使う接続設定."""

    model: str
    host: str = "http://localhost:11434"
    timeout: float = 60.0
    max_workers: int = 1
    cache_enabled: bool = True

    def __post_init__(self) -> None:
        """設定値の妥当性を検証する."""
        if not self.model.strip():
            msg = "ollama_modelは必須です"
            raise ValueError(msg)
        if not self.host.strip():
            msg = "ollama_hostは必須です"
            raise ValueError(msg)
        if self.timeout <= 0:
            msg = f"ollama_timeoutは正の数である必要があります: {self.timeout}"
            raise ValueError(msg)
        if self.max_workers <= 0:
            msg = (
                f"ollama_max_workersは正の整数である必要があります: {self.max_workers}"
            )
            raise ValueError(msg)
