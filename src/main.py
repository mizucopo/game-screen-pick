"""game-screen-pick のCLIエントリポイント."""

import json
import logging
import os
import sys
from importlib import metadata
from pathlib import Path
from typing import Callable, TypeVar

import click

from .application.run_video import run_video_application
from .models.video_run_config import VideoRunConfig
from .models.video_selection_request import (
    MAXIMUM_OUTPUT_COUNT,
    MINIMUM_SAMPLE_INTERVAL_SECONDS,
    VideoSelectionRequest,
)
from .services.game_context_generator import SUPPORTED_GAME_CONTEXT_PROVIDERS
from .utils.elapsed_log_formatter import ElapsedLogFormatter
from .utils.video_run_config_loader import VideoRunConfigLoader

PROJECT_NAME = "game-screen-pick"
SUPPORTED_VIDEO_EXTENSIONS = frozenset(
    {
        ".avi",
        ".flv",
        ".m2ts",
        ".m4v",
        ".mkv",
        ".mov",
        ".mp4",
        ".mpeg",
        ".mpg",
        ".mts",
        ".ts",
        ".webm",
        ".wmv",
    }
)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(ElapsedLogFormatter())
logging.basicConfig(level=logging.INFO, handlers=[console_handler], force=True)
logger = logging.getLogger(__name__)

_ConfigValueT = TypeVar("_ConfigValueT")


def _project_version() -> str:
    """install済みpackage metadataから実行versionを返す."""
    return metadata.version(PROJECT_NAME)


def _display_ollama_host(ollama_host: str | None) -> str:
    """Ollama hostの実効入力値と自動決定元を表示用に返す."""
    if ollama_host:
        return ollama_host
    if "OLLAMA_HOST" in os.environ:
        return f"{os.environ['OLLAMA_HOST']}（自動決定: OLLAMA_HOST）"
    return "127.0.0.1:11434（自動決定: 既定値）"


def _log_cli_start(
    *,
    config_path: str,
    output_count: int,
    game_title: str | None,
    game_context: str,
    game_context_provider: str,
    game_context_model: str | None,
    primary_model: str,
    secondary_model: str,
    ollama_host: str | None,
    ollama_timeout: float,
    allow_cpu: bool,
    ffmpeg_workers: int,
    sample_interval_seconds: float | None,
    debug: bool,
    input_video_dir: str,
    output_dir: str,
) -> None:
    """project情報と実際に適用する実効設定を起動直後に出力する."""
    logger.info("%s %s の画像選定処理を開始します。", PROJECT_NAME, _project_version())
    options: dict[str, object] = {
        "--config": config_path,
        "--num": output_count,
        "--game-title": (
            game_title.strip() if game_title and game_title.strip() else ""
        ),
        "--game-context": game_context.strip(),
        "[run].game_context_provider": game_context_provider,
        "[run].game_context_model": game_context_model or "<provider既定>",
        "[run].primary_model": primary_model,
        "[run].secondary_model": secondary_model,
        "[run].ollama_host": _display_ollama_host(ollama_host),
        "[run].ollama_timeout": ollama_timeout,
        "[run].allow_cpu": allow_cpu,
        "[run].ffmpeg_workers": ffmpeg_workers,
        "[run].sample_interval_seconds": (
            sample_interval_seconds
            if sample_interval_seconds is not None
            else "<自動決定: 動画時間と選択枚数>"
        ),
        "[run].debug": debug,
        "INPUT_VIDEO_DIR": input_video_dir,
        "OUTPUT_DIR": output_dir,
    }
    logger.info("実効設定:")
    for option, value in options.items():
        logger.info("  %s: %s", option, json.dumps(value, ensure_ascii=False))


def validate_positive_int(value: int | str | None) -> int | None:
    """正の整数を検証する."""
    if value is None:
        return None
    try:
        integer_value = int(value)
    except ValueError as error:
        raise click.BadParameter(f"'{value}' は整数ではありません") from error
    if integer_value <= 0:
        raise click.BadParameter(
            f"正の整数を指定してください（実際の値: {integer_value}）"
        )
    return integer_value


def validate_positive_float(value: float | str | None) -> float | None:
    """正の有限浮動小数点数を検証する."""
    if value is None:
        return None
    try:
        float_value = float(value)
    except ValueError as error:
        raise click.BadParameter(f"'{value}' は数値ではありません") from error
    if not 0 < float_value < float("inf"):
        raise click.BadParameter(f"正の数を指定してください（実際の値: {float_value}）")
    return float_value


def validate_output_count(value: int | str | None) -> int | None:
    """選択枚数を対応範囲へ制限する."""
    output_count = validate_positive_int(value)
    if output_count is not None and output_count > MAXIMUM_OUTPUT_COUNT:
        raise click.BadParameter(
            f"{MAXIMUM_OUTPUT_COUNT}以下で指定してください（実際の値: {output_count}）"
        )
    return output_count


def validate_sample_interval(value: float | str | None) -> float | None:
    """候補抽出間隔を実装が保証する下限以上へ制限する."""
    interval = validate_positive_float(value)
    if interval is not None and interval < MINIMUM_SAMPLE_INTERVAL_SECONDS:
        raise click.BadParameter(
            f"{MINIMUM_SAMPLE_INTERVAL_SECONDS}以上で指定してください"
            f"（実際の値: {interval}）"
        )
    return interval


def validate_ffmpeg_workers(value: int | str | None) -> int | None:
    """ffmpeg並列数をCPU負荷を抑える1から4へ制限する."""
    workers = validate_positive_int(value)
    if workers is not None and workers > 4:
        raise click.BadParameter(f"1から4で指定してください（実際の値: {workers}）")
    return workers


def _with_config_hint(
    *,
    key: str,
    resolve: Callable[[], _ConfigValueT],
) -> _ConfigValueT:
    """設定ファイル値の範囲エラーへkey名を付ける."""
    try:
        return resolve()
    except click.BadParameter as error:
        raise click.BadParameter(error.message, param_hint=f"[run].{key}") from error


def resolve_video_run_config(
    *,
    config_path: str,
) -> VideoRunConfig:
    """組み込み既定値とTOMLから実効設定を解決する."""
    try:
        file_values = VideoRunConfigLoader.load(config_path)
    except ValueError as error:
        raise click.BadParameter(str(error), param_hint="--config") from error

    defaults = VideoRunConfig()
    values: dict[str, object] = {
        "game_context_provider": defaults.game_context_provider,
        "game_context_model": defaults.game_context_model,
        "primary_model": defaults.primary_model,
        "secondary_model": defaults.secondary_model,
        "ollama_host": defaults.ollama_host,
        "ollama_timeout": defaults.ollama_timeout,
        "allow_cpu": defaults.allow_cpu,
        "ffmpeg_workers": defaults.ffmpeg_workers,
        "sample_interval_seconds": defaults.sample_interval_seconds,
        "debug": defaults.debug,
    }
    values.update(file_values)

    provider = str(values["game_context_provider"])
    if provider not in SUPPORTED_GAME_CONTEXT_PROVIDERS:
        choices = ", ".join(SUPPORTED_GAME_CONTEXT_PROVIDERS)
        raise click.BadParameter(
            f"{choices}から指定してください（実際の値: {provider}）",
            param_hint="[run].game_context_provider",
        )

    raw_ollama_timeout = values["ollama_timeout"]
    assert isinstance(raw_ollama_timeout, int | float) and not isinstance(
        raw_ollama_timeout, bool
    )
    resolved_ollama_timeout = _with_config_hint(
        key="ollama_timeout",
        resolve=lambda: validate_positive_float(float(raw_ollama_timeout)),
    )
    raw_ffmpeg_workers = values["ffmpeg_workers"]
    assert isinstance(raw_ffmpeg_workers, int) and not isinstance(
        raw_ffmpeg_workers, bool
    )
    resolved_ffmpeg_workers = _with_config_hint(
        key="ffmpeg_workers",
        resolve=lambda: validate_ffmpeg_workers(raw_ffmpeg_workers),
    )
    raw_sample_interval = values["sample_interval_seconds"]
    assert raw_sample_interval is None or (
        isinstance(raw_sample_interval, int | float)
        and not isinstance(raw_sample_interval, bool)
    )
    resolved_sample_interval = _with_config_hint(
        key="sample_interval_seconds",
        resolve=lambda: validate_sample_interval(
            float(raw_sample_interval) if raw_sample_interval is not None else None
        ),
    )
    assert isinstance(resolved_ollama_timeout, float)
    assert isinstance(resolved_ffmpeg_workers, int)
    assert resolved_sample_interval is None or isinstance(
        resolved_sample_interval, float
    )

    return VideoRunConfig(
        game_context_provider=provider,
        game_context_model=(
            str(values["game_context_model"])
            if values["game_context_model"] is not None
            else None
        ),
        primary_model=str(values["primary_model"]),
        secondary_model=str(values["secondary_model"]),
        ollama_host=(
            str(values["ollama_host"]) if values["ollama_host"] is not None else None
        ),
        ollama_timeout=resolved_ollama_timeout,
        allow_cpu=bool(values["allow_cpu"]),
        ffmpeg_workers=resolved_ffmpeg_workers,
        sample_interval_seconds=resolved_sample_interval,
        debug=bool(values["debug"]),
    )


def discover_input_videos(input_video_dir: str) -> tuple[str, ...]:
    """入力ディレクトリ直下の対象動画を安定した順序で列挙する."""
    input_path = Path(input_video_dir)
    if not input_path.is_dir():
        raise click.BadParameter(
            f"入力動画ディレクトリが見つかりません: {input_video_dir}",
            param_hint="INPUT_VIDEO_DIR",
        )
    videos = tuple(
        str(path)
        for path in sorted(input_path.iterdir(), key=lambda path: path.name)
        if not path.is_symlink()
        and path.is_file()
        and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
    )
    if not videos:
        extensions = ", ".join(sorted(SUPPORTED_VIDEO_EXTENSIONS))
        raise click.BadParameter(
            f"処理対象の動画が見つかりません: {input_video_dir}"
            f"（対応拡張子: {extensions}）",
            param_hint="INPUT_VIDEO_DIR",
        )
    return videos


def validate_game_context_input(
    game_title: str | None,
    game_context: str,
) -> None:
    """Game TitleとGame Contextを常にXORへ制限する."""
    has_title = bool(game_title and game_title.strip())
    has_context = bool(game_context.strip())
    if has_title and has_context:
        raise click.UsageError(
            "--game-titleと--game-contextのどちらか一方だけを指定してください"
        )
    if not has_title and not has_context:
        raise click.UsageError(
            "--game-titleと--game-contextのどちらか一方を指定してください"
        )


@click.command()
@click.option(
    "-c",
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    default="config.toml",
    show_default=True,
    help="TOML設定ファイル",
)
@click.option(
    "-n",
    "--num",
    "output_count",
    required=True,
    type=int,
    callback=lambda _ctx, _param, value: validate_output_count(value),
    help=f"選択枚数（1から{MAXIMUM_OUTPUT_COUNT}）",
)
@click.option(
    "--game-title",
    default=None,
    help="Web検索からGame Contextを生成するためのゲーム表記",
)
@click.option(
    "--game-context",
    default="",
    help="画像選定に直接使うGame Context",
)
@click.argument("input_video_dir", type=click.Path(path_type=str))
@click.argument("output_dir", type=click.Path(path_type=str))
def execute(
    config_path: str,
    output_count: int,
    game_title: str | None,
    game_context: str,
    input_video_dir: str,
    output_dir: str,
) -> None:
    """入力ディレクトリのゲーム動画全体からブログ掲載用画像を選定する."""
    config = resolve_video_run_config(config_path=config_path)
    input_videos = discover_input_videos(input_video_dir)
    validate_game_context_input(game_title, game_context)
    _log_cli_start(
        config_path=config_path,
        output_count=output_count,
        game_title=game_title,
        game_context=game_context,
        game_context_provider=config.game_context_provider,
        game_context_model=config.game_context_model,
        primary_model=config.primary_model,
        secondary_model=config.secondary_model,
        ollama_host=config.ollama_host,
        ollama_timeout=config.ollama_timeout,
        allow_cpu=config.allow_cpu,
        ffmpeg_workers=config.ffmpeg_workers,
        sample_interval_seconds=config.sample_interval_seconds,
        debug=config.debug,
        input_video_dir=input_video_dir,
        output_dir=output_dir,
    )
    run_video_application(
        VideoSelectionRequest(
            input_videos=input_videos,
            output_dir=output_dir,
            output_count=output_count,
            game_title=game_title,
            game_context=game_context,
            game_context_provider=config.game_context_provider,
            game_context_model=config.game_context_model,
            primary_model=config.primary_model,
            secondary_model=config.secondary_model,
            ollama_host=config.ollama_host,
            ollama_timeout=config.ollama_timeout,
            allow_cpu=config.allow_cpu,
            ffmpeg_workers=config.ffmpeg_workers,
            sample_interval_seconds=config.sample_interval_seconds,
            debug=config.debug,
        )
    )


def run(args: list[str]) -> None:
    """引数配列を使ってCLIを実行する."""
    original_argv = sys.argv
    try:
        sys.argv = ["game-screen-pick", *args]
        execute(standalone_mode=False)
    except click.ClickException as error:
        error.show()
        raise SystemExit(error.exit_code) from error
    finally:
        sys.argv = original_argv


def cli_main() -> None:
    """project script用の薄いentrypoint."""
    run(sys.argv[1:])


if __name__ == "__main__":
    cli_main()
