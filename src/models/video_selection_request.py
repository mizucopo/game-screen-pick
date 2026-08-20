"""単一動画の画像選定リクエスト."""

from dataclasses import dataclass

MAXIMUM_OUTPUT_COUNT = 600
MINIMUM_SAMPLE_INTERVAL_SECONDS = 0.25


@dataclass(frozen=True)
class VideoSelectionRequest:
    """CLIから単一動画の画像選定へ渡すリクエスト."""

    input_video: str
    output_dir: str
    output_count: int
    game_title: str | None
    game_context: str
    primary_model: str
    secondary_model: str
    ollama_host: str | None
    ollama_timeout: float
    allow_cpu: bool
    ffmpeg_workers: int
    sample_interval_seconds: float | None
    debug: bool
