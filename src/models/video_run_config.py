"""動画選定CLIの実効設定."""

from dataclasses import dataclass


@dataclass(frozen=True)
class VideoRunConfig:
    """設定ファイルとCLI optionを統合した実効設定."""

    output_count: int = 30
    game_title: str | None = None
    game_context: str = ""
    game_context_provider: str = "ollama"
    game_context_model: str | None = None
    primary_model: str = "qwen3.8:27b"
    secondary_model: str = "muse-glimmer:30b"
    ollama_host: str | None = None
    ollama_timeout: float = 900.0
    allow_cpu: bool = False
    ffmpeg_workers: int = 2
    sample_interval_seconds: float | None = None
    debug: bool = False
