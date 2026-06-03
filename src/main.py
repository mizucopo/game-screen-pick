"""game-screen-pick のCLIエントリポイント."""

import logging
import sys

import click

from .application.run import run_application
from .models.application_run_request import ApplicationRunRequest

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    stream=sys.stdout,
    force=True,
)


def validate_positive_int(value: str | None) -> int | None:
    """正の整数をバリデーションする.

    Args:
        value: CLIから渡された文字列値。

    Returns:
        検証済みの整数値。 `None` が渡された場合は `None` を返す。

    Raises:
        click.BadParameter: 整数でない、または1未満の値が渡された場合。
    """
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


def validate_positive_int_or_zero(value: str | None) -> int | None:
    """0または正の整数をバリデーションする.

    CLI上では「0でシングルスレッド」を許可するが、
    `ThreadPoolExecutor(max_workers=0)` は実行時エラーになるため、
    内部では0を1へ正規化して返す。

    Args:
        value: CLIから渡された文字列値。

    Returns:
        検証済みの整数値。 `0` は `1` に変換される。

    Raises:
        click.BadParameter: 整数でない、または負の値が渡された場合。
    """
    if value is None:
        return None
    try:
        integer_value = int(value)
    except ValueError as error:
        raise click.BadParameter(f"'{value}' は整数ではありません") from error
    if integer_value < 0:
        raise click.BadParameter(
            f"0以上の整数を指定してください（実際の値: {integer_value}）"
        )
    return integer_value if integer_value > 0 else 1


def validate_similarity_range(value: float | str | None) -> float | None:
    """類似度しきい値をバリデーションする.

    Args:
        value: CLIから渡された文字列値。

    Returns:
        0.0以上1.0以下で検証済みの浮動小数点値。
        `None` が渡された場合は `None` を返す。

    Raises:
        click.BadParameter: 数値でない、または許容範囲外の値の場合。
    """
    if value is None:
        return None
    try:
        float_value = float(value)
    except ValueError as error:
        raise click.BadParameter(f"'{value}' は数値ではありません") from error
    if not 0.0 <= float_value <= 1.0:
        raise click.BadParameter(
            f"0.0~1.0の範囲で指定してください（実際の値: {float_value}）"
        )
    return float_value


def validate_positive_float(value: float | str | None) -> float | None:
    """正の浮動小数点数をバリデーションする."""
    if value is None:
        return None
    try:
        float_value = float(value)
    except ValueError as error:
        raise click.BadParameter(f"'{value}' は数値ではありません") from error
    if float_value <= 0:
        raise click.BadParameter(f"正の数を指定してください（実際の値: {float_value}）")
    return float_value


@click.command()
@click.option(
    "-n",
    "--num",
    default=100,
    type=int,
    callback=lambda _ctx, _param, x: validate_positive_int(x),
    help="選択枚数",
)
@click.option(
    "-s",
    "--similarity",
    default=None,
    type=float,
    callback=lambda _ctx, _param, x: validate_similarity_range(x),
    help="類似度しきい値(0.7~0.85推奨)",
)
@click.option("-r", "--recursive", is_flag=True, help="サブフォルダも検索")
@click.option(
    "--profile",
    type=click.Choice(["auto", "active", "static"]),
    default=None,
    help="選定プロファイル",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    default=None,
    help="TOML設定ファイル",
)
@click.option(
    "--ollama-model",
    default=None,
    type=str,
    help="Ollamaの画像分類モデル名",
)
@click.option(
    "--ollama-host",
    default=None,
    type=str,
    help="OllamaホストURL（OLLAMA_HOSTより優先）",
)
@click.option(
    "--ollama-timeout",
    type=float,
    callback=lambda _ctx, _param, x: validate_positive_float(x),
    default=None,
    help="Ollama APIタイムアウト秒数",
)
@click.option(
    "--ollama-max-workers",
    type=int,
    callback=lambda _ctx, _param, x: validate_positive_int(x),
    default=None,
    help="Ollama分類の並列ワーカー数",
)
@click.option(
    "--no-ollama-cache",
    "ollama_cache_enabled",
    flag_value=False,
    default=True,
    help="Ollama分類キャッシュを使わない",
)
@click.option(
    "--scene-hint",
    default=None,
    type=str,
    help="scene catalog作成に渡す任意ヒント",
)
@click.option(
    "--report-json",
    type=click.Path(dir_okay=False, path_type=str),
    default=None,
    help="JSONレポートの出力先",
)
@click.option(
    "--rename",
    is_flag=True,
    help="scene別に play0001.ext / event0001.ext 形式で出力ファイル名を付け直す",
)
@click.option(
    "--batch-size",
    type=int,
    callback=lambda _ctx, _param, x: validate_positive_int(x),
    default=None,
    help="CLIP推論のバッチサイズ",
)
@click.option(
    "--result-max-workers",
    type=int,
    callback=lambda _ctx, _param, x: validate_positive_int_or_zero(x),
    default=None,
    help="結果構築の並列ワーカー数（0でシングルスレッド）",
)
@click.option(
    "--max-dim",
    type=int,
    callback=lambda _ctx, _param, x: validate_positive_int(x),
    default=720,
    help="画像リサイズ時の長辺の最大ピクセル数",
)
@click.option(
    "--max-memory-gb",
    type=int,
    callback=lambda _ctx, _param, x: validate_positive_int(x),
    default=1,
    help="チャンク処理時のメモリ予算（GB）",
)
@click.option("--debug", is_flag=True, help="デバッグログを有効化")
@click.argument(
    "input_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
)
@click.argument(
    "output_dir",
    type=click.Path(file_okay=False, dir_okay=True),
)
def execute(
    num: int,
    similarity: float | None,
    recursive: bool,
    profile: str | None,
    config_path: str | None,
    ollama_model: str | None,
    ollama_host: str | None,
    ollama_timeout: float | None,
    ollama_max_workers: int | None,
    ollama_cache_enabled: bool,
    scene_hint: str | None,
    report_json: str | None,
    rename: bool,
    batch_size: int | None,
    result_max_workers: int | None,
    max_dim: int,
    max_memory_gb: int,
    debug: bool,
    input_dir: str,
    output_dir: str,
) -> None:
    """ゲーム画面からscene mixを保って画像を選択する.

    CLIはオプション変換と入力検証に集中し、application実行層へ
    リクエストを渡す。

    \b
    使用例:
      game-screen-pick -n 15 ./screenshots ./output
      game-screen-pick --rename ./screenshots ./output
      game-screen-pick --ollama-model gemma4 --scene-hint "RPG" ./in ./out

    Args:
        num: 選択枚数。
        similarity: 類似度しきい値。未指定時は設定ファイルまたは既定値を使う。
        recursive: サブフォルダを再帰的に探索するかどうか。
        profile: 選定プロファイル。 `auto` / `active` / `static` 。
        config_path: TOML設定ファイルのパス。
        ollama_model: Ollamaの画像分類モデル名。
        ollama_host: OllamaホストURL。
        ollama_timeout: Ollama APIタイムアウト秒数。
        ollama_max_workers: Ollama分類の並列ワーカー数。
        ollama_cache_enabled: Ollama分類キャッシュを使うかどうか。
        scene_hint: scene catalog作成に渡す任意ヒント。
        report_json: JSONレポートの出力先パス。
        rename: scene別の連番ファイル名で出力するかどうか。
        batch_size: CLIP推論のバッチサイズ上書き。
        result_max_workers: 結果構築に使う並列ワーカー数。
        max_dim: 入力画像の長辺最大サイズ。
        max_memory_gb: チャンク処理のメモリ予算。
        debug: デバッグログを有効化するかどうか。
        input_dir: 入力画像フォルダ。
        output_dir: 選択画像のコピー先フォルダ。

    Returns:
        なし。

    Raises:
        click.BadParameter: 入力値または入力パスが不正な場合。
        SystemExit: 想定外の実行時エラーをCLI終了コードへ変換する場合。
    """
    run_application(
        ApplicationRunRequest(
            num=num,
            similarity=similarity,
            recursive=recursive,
            profile=profile,
            config_path=config_path,
            ollama_model=ollama_model,
            ollama_host=ollama_host,
            ollama_timeout=ollama_timeout,
            ollama_max_workers=ollama_max_workers,
            ollama_cache_enabled=ollama_cache_enabled,
            scene_hint=scene_hint,
            report_json=report_json,
            rename=rename,
            batch_size=batch_size,
            result_max_workers=result_max_workers,
            max_dim=max_dim,
            max_memory_gb=max_memory_gb,
            debug=debug,
            input_dir=input_dir,
            output_dir=output_dir,
        )
    )


def run(args: list[str]) -> None:
    """CLIを実行する.

    `args` を `sys.argv` へ一時反映し、Clickフローで `execute` を呼び出す。

    Returns:
        なし。

    Raises:
        SystemExit: 実行時エラーを終了コードへ変換した場合。
    """
    original_argv = sys.argv
    try:
        sys.argv = ["game-screen-pick"] + args
        execute(standalone_mode=False)
    except click.ClickException as error:
        error.show()
        raise SystemExit(error.exit_code) from error
    finally:
        sys.argv = original_argv


def cli_main() -> None:
    """CLIエントリポイント関数.

    `pyproject.toml` の script entrypoint から呼ばれる薄いラッパーで、
    `run` を生成して実行するだけに責務を限定する。
    """
    run(sys.argv[1:])


if __name__ == "__main__":
    cli_main()
