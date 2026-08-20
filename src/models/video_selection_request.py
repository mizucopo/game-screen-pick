"""1本以上の動画を扱う画像選定リクエスト."""

from dataclasses import InitVar, dataclass

MAXIMUM_OUTPUT_COUNT = 600
MINIMUM_SAMPLE_INTERVAL_SECONDS = 0.25


@dataclass(frozen=True)
class VideoSelectionRequest:
    """CLIから動画の画像選定へ渡すリクエスト."""

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
    input_videos: tuple[str, ...] = ()
    input_video: InitVar[str | None] = None

    def __post_init__(self, input_video: str | None) -> None:
        """旧単一入力constructorを正規化し、1本以上を保証する."""
        if input_video is not None:
            if self.input_videos:
                raise ValueError("input_videoとinput_videosは同時に指定できません")
            object.__setattr__(self, "input_videos", (input_video,))
        if not self.input_videos:
            raise ValueError("入力動画を1本以上指定してください")
