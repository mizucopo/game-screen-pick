"""game_screen_pick - 高精度・コンテンツ多様性重視選択ツール."""

import logging
import random
import sys
from pathlib import Path

import click

from .analyzers.image_quality_analyzer import ImageQualityAnalyzer
from .models.analyzer_config import AnalyzerConfig
from .models.selection_config import SelectionConfig
from .services.game_screen_picker import GameScreenPicker
from .utils.file_utils import FileUtils
from .utils.result_formatter import ResultFormatter

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    stream=sys.stdout,
    force=True,
)

logger = logging.getLogger(__name__)


class Main:
    """CLIメインクラス."""

    def __init__(self, args: list[str] | None = None):
        """Mainクラスを初期化する.

        Args:
            args: コマンドライン引数リスト（テスト用）
        """
        self.args = args

    @staticmethod
    def validate_positive_int(value: str | None) -> int | None:
        """正の整数をバリデーションする.

        Args:
            value: コマンドラインから渡された文字列値

        Returns:
            バリデーション済みの整数値

        Raises:
            click.BadParameter: 値が正の整数でない場合
        """
        if value is None:
            return None
        try:
            ivalue = int(value)
            if ivalue <= 0:
                raise click.BadParameter(
                    f"正の整数を指定してください（実際の値: {ivalue}）"
                )
            return ivalue
        except ValueError as e:
            raise click.BadParameter(f"'{value}' は整数ではありません") from e

    @staticmethod
    def validate_positive_int_or_zero(value: str | None) -> int | None:
        """0または正の整数をバリデーションする.

        Note:
            ThreadPoolExecutor(max_workers=0) は実行時エラーになるため、
            0 は 1 に変換して返す（CLIヘルプの「0でシングルスレッド」
            という仕様を維持しつつ、実行時エラーを回避）。

        Args:
            value: コマンドラインから渡された文字列値

        Returns:
            バリデーション済みの整数値（0は1に変換）

        Raises:
            click.BadParameter: 値が0以上の整数でない場合
        """
        if value is None:
            return None
        try:
            ivalue = int(value)
            if ivalue < 0:
                raise click.BadParameter(
                    f"0以上の整数を指定してください（実際の値: {ivalue}）"
                )
            # ThreadPoolExecutor(max_workers=0) は ValueError になるため
            # 0 を指定した場合は 1 (シングルスレッド) に変換
            return ivalue if ivalue > 0 else 1
        except ValueError as e:
            raise click.BadParameter(f"'{value}' は整数ではありません") from e

    @staticmethod
    def validate_similarity_range(value: str | None) -> float | None:
        """類似度しきい値をバリデーションする（0.0 ~ 1.0）.

        Args:
            value: コマンドラインから渡された文字列値

        Returns:
            バリデーション済みの浮動小数点値

        Raises:
            click.BadParameter: 値が0.0~1.0の範囲外の場合
        """
        if value is None:
            return None
        try:
            fvalue = float(value)
            if not 0.0 <= fvalue <= 1.0:
                raise click.BadParameter(
                    f"0.0~1.0の範囲で指定してください（実際の値: {fvalue}）"
                )
            return fvalue
        except ValueError as e:
            raise click.BadParameter(f"'{value}' は数値ではありません") from e

    @click.command()
    @click.option(
        "-n",
        "--num",
        default=10,
        type=int,
        callback=lambda _ctx, _param, x: Main.validate_positive_int(x),
        help="選択枚数",
    )
    @click.option(
        "-s",
        "--similarity",
        default=0.72,
        type=float,
        callback=lambda _ctx, _param, x: Main.validate_similarity_range(x),
        help="類似度しきい値(0.7~0.85推奨)",
    )
    @click.option("-r", "--recursive", is_flag=True, help="サブフォルダも検索")
    @click.option(
        "--seed",
        type=int,
        default=None,
        help="乱数シード（再現可能な結果を得るために指定）",
    )
    @click.option(
        "--batch-size",
        type=int,
        callback=lambda _ctx, _param, x: Main.validate_positive_int(x),
        default=None,
        help=(
            "CLIP推論のバッチサイズ（デフォルト: 32、大きいほど高速だがメモリ消費増加）"
        ),
    )
    @click.option(
        "--result-max-workers",
        type=int,
        callback=lambda _ctx, _param, x: Main.validate_positive_int_or_zero(x),
        default=None,
        help="結果構築の並列ワーカー数（デフォルト: 自動設定、0でシングルスレッド）",
    )
    @click.option(
        "--max-dim",
        type=int,
        callback=lambda _ctx, _param, x: Main.validate_positive_int(x),
        default=720,
        help=(
            "画像リサイズ時の長辺の最大ピクセル数"
            "（デフォルト: 720、小さいほど高速だが精度低下）"
        ),
    )
    @click.option(
        "--max-memory-mb",
        type=int,
        callback=lambda _ctx, _param, x: Main.validate_positive_int(x),
        default=512,
        help=(
            "チャンク処理時のメモリ予算（MB）"
            "(デフォルト: 512、大きいほどチャンクが大きくなる）"
        ),
    )
    @click.option("--debug", is_flag=True, help="デバッグログを有効化")
    @click.argument(
        "input", type=click.Path(exists=True, file_okay=False, dir_okay=True)
    )
    @click.argument("output", type=click.Path(file_okay=False, dir_okay=True))
    def _execute(
        num: int,
        similarity: float,
        recursive: bool,
        seed: int | None,
        batch_size: int | None,
        result_max_workers: int | None,
        max_dim: int,
        max_memory_mb: int,
        debug: bool,
        input: str,
        output: str,
    ) -> None:
        """ゲームスクリーンショットから品質・多様性を考慮して最適な画像を自動選択するAIツール.

        \b
        使用例:
          game-screen-pick -n 15 ./screenshots ./output
          game-screen-pick -r -n 10 ./screenshots ./output
        """
        # デバッグモードの場合はログレベルをDEBUGに設定
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)

        try:
            # 入力フォルダのパスを検証
            input_path = Path(input)
            if not input_path.is_dir():
                raise click.BadParameter(
                    f"指定パスはフォルダではありません: {input}", param_hint="input"
                )

            # 依存関係の遅延初期化
            analyzer_config = AnalyzerConfig.from_cli_args(
                result_max_workers=result_max_workers,
                max_dim=max_dim,
                max_memory_mb=max_memory_mb,
            )
            analyzer = ImageQualityAnalyzer(config=analyzer_config)

            rng = random.Random(seed) if seed is not None else None

            selection_config = SelectionConfig.from_cli_args(batch_size=batch_size)
            picker = GameScreenPicker(analyzer, config=selection_config, rng=rng)

            best, stats = picker.select(
                str(input_path),
                num,
                similarity,
                recursive,
            )

            FileUtils.copy_selected_items(best, output)
            ResultFormatter.display_results(best, stats)

        except Exception as e:
            import traceback

            logger.error(f"予期しないエラーが発生しました: {type(e).__name__}: {e}")
            logger.error(traceback.format_exc())
            raise SystemExit(1) from None

    def run(self) -> None:
        """CLIを実行する.

        Raises:
            click.ClickException: バリデーションエラー時
        """
        if self.args is not None:
            # テスト時はsys.argvを差し替えてclickを実行
            original_argv = sys.argv
            try:
                sys.argv = ["game-screen-pick"] + self.args
                self._execute(standalone_mode=False)
            finally:
                sys.argv = original_argv
        else:
            self._execute(standalone_mode=False)


def cli_main() -> None:
    """CLIエントリポイント関数.

    pyproject.tomlの[project.scripts]から呼び出される。
    """
    Main().run()


if __name__ == "__main__":
    cli_main()
