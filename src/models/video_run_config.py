"""動画選定CLIの実効設定."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class VideoRunConfig:
    """設定ファイルから読み込む実効設定."""

    game_context_provider: str | None = None
    game_context_model: str | None = None
    ollama_api_key: str | None = field(default=None, repr=False)
    openai_api_key: str | None = field(default=None, repr=False)
    gemini_api_key: str | None = field(default=None, repr=False)
    xai_api_key: str | None = field(default=None, repr=False)
    primary_model: str = "qwen3.8:27b"
    secondary_model: str = "muse-glimmer:30b"
    ollama_host: str | None = None
    ollama_timeout: float = 900.0
    allow_cpu: bool = False
    ffmpeg_workers: int = 2
    sample_interval_seconds: float | None = None
    debug: bool = False
