"""Ollama接続設定."""

from dataclasses import dataclass


@dataclass(frozen=True)
class OllamaConfig:
    """Ollama scene分類に使う接続設定."""

    DEFAULT_PORT = "11434"

    model: str
    host: str = "http://localhost:11434"
    timeout: float = 60.0
    max_workers: int = 1
    cache_enabled: bool = True

    def __post_init__(self) -> None:
        """設定値の妥当性を検証する."""
        normalized_host = self._normalize_host(self.host)
        if not self.model.strip():
            msg = "ollama_modelは必須です"
            raise ValueError(msg)
        if not normalized_host:
            msg = "ollama_hostは必須です"
            raise ValueError(msg)
        object.__setattr__(self, "host", normalized_host)
        if self.timeout <= 0:
            msg = f"ollama_timeoutは正の数である必要があります: {self.timeout}"
            raise ValueError(msg)
        if self.max_workers <= 0:
            msg = (
                f"ollama_max_workersは正の整数である必要があります: {self.max_workers}"
            )
            raise ValueError(msg)

    @classmethod
    def _normalize_host(cls, host: str) -> str:
        """schemeなしのhostへHTTP schemeと既定portを補完する."""
        stripped_host = host.strip()
        if not stripped_host:
            return stripped_host
        if "://" in stripped_host:
            return stripped_host
        return f"http://{cls._with_default_port(stripped_host)}"

    @classmethod
    def _with_default_port(cls, host: str) -> str:
        """portなしのhostへOllama既定portを補完する."""
        if host.startswith("["):
            return host if "]:" in host else f"{host}:{cls.DEFAULT_PORT}"
        if ":" in host:
            return host
        return f"{host}:{cls.DEFAULT_PORT}"
