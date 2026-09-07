"""vLLMのOpenAI互換endpointに使う接続設定."""

import math
from dataclasses import dataclass, field
from urllib.parse import urlsplit, urlunsplit


@dataclass(frozen=True)
class VllmConfig:
    """動画理解modelと利用者管理のcache revisionを保持する."""

    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = ""
    api_key: str | None = field(default=None, repr=False)
    timeout_seconds: float = 900.0
    cache_revision: str = "1"

    def __post_init__(self) -> None:
        """値を検証し、endpointとmodel名を正規化する."""
        if not self.model.strip():
            raise ValueError("vllm_modelは必須です")
        if not self.cache_revision.strip():
            raise ValueError("vllm_cache_revisionは必須です")
        if (
            isinstance(self.timeout_seconds, bool)
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("vllm_timeout_secondsは有限の正の数が必要です")
        try:
            parsed = urlsplit(self.base_url.strip())
            valid_port = parsed.port is None or 0 < parsed.port <= 65535
        except ValueError:
            raise ValueError("vllm_base_urlが不正です") from None
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or not valid_port
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("vllm_base_urlには認証情報なしのHTTP(S) URLが必要です")
        path = parsed.path.rstrip("/")
        if not path.endswith("/v1"):
            path += "/v1"
        object.__setattr__(self, "base_url", urlunsplit(parsed._replace(path=path)))
        object.__setattr__(self, "model", self.model.strip())
        object.__setattr__(self, "cache_revision", self.cache_revision.strip())
