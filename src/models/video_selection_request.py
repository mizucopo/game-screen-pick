"""1本以上の動画を扱う画像選定リクエスト."""

from dataclasses import dataclass, field

MAXIMUM_OUTPUT_COUNT = 999
MINIMUM_SAMPLE_INTERVAL_SECONDS = 0.25
_MISSING = object()


@dataclass(frozen=True, init=False)
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
    input_videos: tuple[str, ...]
    game_context_provider: str | None
    game_context_model: str | None
    game_context_api_key: str | None = field(default=None, repr=False)

    def __init__(
        self,
        input_video: str | None = None,
        output_dir: str | object = _MISSING,
        output_count: int | object = _MISSING,
        game_title: str | None | object = _MISSING,
        game_context: str | object = _MISSING,
        primary_model: str | object = _MISSING,
        secondary_model: str | object = _MISSING,
        ollama_host: str | None | object = _MISSING,
        ollama_timeout: float | object = _MISSING,
        allow_cpu: bool | object = _MISSING,
        ffmpeg_workers: int | object = _MISSING,
        sample_interval_seconds: float | None | object = _MISSING,
        debug: bool | object = _MISSING,
        *,
        input_videos: tuple[str, ...] = (),
        game_context_provider: str | None = None,
        game_context_model: str | None = None,
        game_context_api_key: str | None = None,
    ) -> None:
        """旧位置指定と新しい複数入力keywordを同じrequestへ正規化する."""
        values = {
            "output_dir": output_dir,
            "output_count": output_count,
            "game_title": game_title,
            "game_context": game_context,
            "primary_model": primary_model,
            "secondary_model": secondary_model,
            "ollama_host": ollama_host,
            "ollama_timeout": ollama_timeout,
            "allow_cpu": allow_cpu,
            "ffmpeg_workers": ffmpeg_workers,
            "sample_interval_seconds": sample_interval_seconds,
            "debug": debug,
        }
        missing = [name for name, value in values.items() if value is _MISSING]
        if missing:
            raise TypeError(f"必須引数が不足しています: {', '.join(missing)}")
        if input_video is not None:
            if len(input_videos) > 1:
                raise ValueError("input_videoとinput_videosは同時に指定できません")
            input_videos = (input_video,)
        if not input_videos:
            raise ValueError("入力動画を1本以上指定してください")
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "input_videos", tuple(input_videos))
        object.__setattr__(self, "game_context_provider", game_context_provider)
        object.__setattr__(self, "game_context_model", game_context_model)
        object.__setattr__(self, "game_context_api_key", game_context_api_key)

    @property
    def input_video(self) -> str | None:
        """旧単一入力属性をread-onlyで公開し、複数入力ではNoneを返す."""
        if len(self.input_videos) != 1:
            return None
        return self.input_videos[0]
