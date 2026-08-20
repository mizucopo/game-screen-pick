"""game-screen-pick のCLIエントリポイント."""

import logging
import sys
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

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(ElapsedLogFormatter())
logging.basicConfig(level=logging.INFO, handlers=[console_handler], force=True)


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
@click.argument(
    "paths",
    nargs=-1,
    required=True,
    type=click.Path(path_type=str),
    metavar="INPUT_VIDEO... OUTPUT_DIR",
)
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
    paths: tuple[str, ...],
) -> None:
    """1本以上のゲーム動画全体からブログ掲載用画像を選定する."""
    if len(paths) < 2:
        raise click.UsageError("入力動画を1本以上と出力フォルダを指定してください")
    input_videos = paths[:-1]
    output_dir = paths[-1]
    for input_video in input_videos:
        if not Path(input_video).is_file():
            raise click.BadParameter(
                f"入力動画が見つかりません: {input_video}",
                param_hint="INPUT_VIDEOS",
            )
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
