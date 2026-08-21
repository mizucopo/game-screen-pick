"""game-screen-pick のCLIエントリポイント."""

import json
import logging
import os
import sys
from importlib import metadata
from pathlib import Path

import click

from .application.run_video import run_video_application
from .models.video_selection_request import (
    MAXIMUM_OUTPUT_COUNT,
    MINIMUM_SAMPLE_INTERVAL_SECONDS,
    VideoSelectionRequest,
)
from .utils.elapsed_log_formatter import ElapsedLogFormatter

DEFAULT_PRIMARY_MODEL = "qwen3.8:27b"
DEFAULT_SECONDARY_MODEL = "muse-glimmer:30b"
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
    output_count: int,
    game_title: str | None,
    game_context: str,
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
    """project情報と実際に適用するCLI optionを起動直後に出力する."""
    logger.info("%s %s の画像選定処理を開始します。", PROJECT_NAME, _project_version())
    options: dict[str, object] = {
        "--num": output_count,
        "--game-title": (
            game_title.strip()
            if game_title and game_title.strip()
            else "<自動決定: 動画ファイル名>"
        ),
        "--game-context": game_context.strip(),
        "--primary-model": primary_model,
        "--secondary-model": secondary_model,
        "--ollama-host": _display_ollama_host(ollama_host),
        "--ollama-timeout": ollama_timeout,
        "--allow-cpu": allow_cpu,
        "--ffmpeg-workers": ffmpeg_workers,
        "--sample-interval-seconds": (
            sample_interval_seconds
            if sample_interval_seconds is not None
            else "<自動決定: 動画時間と選択枚数>"
        ),
        "--debug": debug,
        "INPUT_VIDEO_DIR": input_video_dir,
        "OUTPUT_DIR": output_dir,
    }
    logger.info("起動オプション:")
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
    """選択枚数をcontact sheetがJPEGに収まる範囲へ制限する."""
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


@click.command()
@click.option(
    "-n",
    "--num",
    "output_count",
    default=30,
    show_default=True,
    type=int,
    callback=lambda _ctx, _param, value: validate_output_count(value),
    help=f"選択枚数（1から{MAXIMUM_OUTPUT_COUNT}）",
)
@click.option(
    "--game-title",
    default=None,
    help="ゲームタイトル。未指定時は動画ファイル名から推測",
)
@click.option(
    "--game-context",
    default="",
    help="ゲーム内容や掲載意図の任意補足",
)
@click.option(
    "--primary-model",
    default=DEFAULT_PRIMARY_MODEL,
    show_default=True,
    help="一次評価に使うOllama vision model",
)
@click.option(
    "--secondary-model",
    default=DEFAULT_SECONDARY_MODEL,
    show_default=True,
    help="遷移確認を含む二次評価に使うOllama vision model",
)
@click.option(
    "--ollama-host",
    default=None,
    help="Ollama host。未指定時はOLLAMA_HOST、その後localhostを使用",
)
@click.option(
    "--ollama-timeout",
    default=900.0,
    show_default=True,
    type=float,
    callback=lambda _ctx, _param, value: validate_positive_float(value),
    help="Ollama APIのbatch単位timeout秒数",
)
@click.option(
    "--allow-cpu",
    is_flag=True,
    help="Ollama modelのGPU利用を確認できなくても続行",
)
@click.option(
    "--ffmpeg-workers",
    default=2,
    show_default=True,
    type=int,
    callback=lambda _ctx, _param, value: validate_ffmpeg_workers(value),
    help="候補フレーム抽出の並列数（1から4）",
)
@click.option(
    "--sample-interval-seconds",
    default=None,
    type=float,
    callback=lambda _ctx, _param, value: validate_sample_interval(value),
    help=(
        f"候補抽出の最大間隔（{MINIMUM_SAMPLE_INTERVAL_SECONDS}秒以上）。"
        "未指定時は動画時間と選択枚数から自動決定"
    ),
)
@click.option("--debug", is_flag=True, help="デバッグログを有効化")
@click.argument("input_video_dir", type=click.Path(path_type=str))
@click.argument("output_dir", type=click.Path(path_type=str))
def execute(
    output_count: int,
    game_title: str | None,
    game_context: str,
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
    """入力ディレクトリのゲーム動画全体からブログ掲載用画像を選定する."""
    _log_cli_start(
        output_count=output_count,
        game_title=game_title,
        game_context=game_context,
        primary_model=primary_model,
        secondary_model=secondary_model,
        ollama_host=ollama_host,
        ollama_timeout=ollama_timeout,
        allow_cpu=allow_cpu,
        ffmpeg_workers=ffmpeg_workers,
        sample_interval_seconds=sample_interval_seconds,
        debug=debug,
        input_video_dir=input_video_dir,
        output_dir=output_dir,
    )
    input_videos = discover_input_videos(input_video_dir)
    run_video_application(
        VideoSelectionRequest(
            input_videos=input_videos,
            output_dir=output_dir,
            output_count=output_count,
            game_title=game_title,
            game_context=game_context,
            primary_model=primary_model,
            secondary_model=secondary_model,
            ollama_host=ollama_host,
            ollama_timeout=ollama_timeout,
            allow_cpu=allow_cpu,
            ffmpeg_workers=ffmpeg_workers,
            sample_interval_seconds=sample_interval_seconds,
            debug=debug,
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
