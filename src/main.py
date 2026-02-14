"""game_screen_pick - 高精度・コンテンツ多様性重視選択ツール."""

import argparse
import logging
import random
import sys
from pathlib import Path

from .analyzers.image_quality_analyzer import ImageQualityAnalyzer
from .models.analyzer_config import AnalyzerConfig
from .models.selection_config import SelectionConfig
from .services.game_screen_picker import GameScreenPicker
from .utils.file_utils import FileUtils
from .utils.result_formatter import ResultFormatter

logger = logging.getLogger(__name__)


class Main:
    """CLIメインクラス.

    画像選択ツールのコマンドラインインターフェースを提供する。
    引数パース、バリデーション、実行の順に処理を行う。
    """

    def __init__(
        self,
        analyzer: ImageQualityAnalyzer | None = None,
        picker: GameScreenPicker | None = None,
        args: list[str] | None = None,
    ):
        """Mainクラスを初期化する.

        Args:
            analyzer: 画像品質アナライザー（Noneの場合はデフォルト生成）
            picker: 画面選択ピッカー（Noneの場合はデフォルト生成）
            args: コマンドライン引数リスト（Noneの場合はsys.argvを使用）
        """
        self.args = args
        self._analyzer = analyzer
        self._picker = picker

    def run(self) -> None:
        """CLIを実行する."""
        parsed_args = self._parse_arguments()

        # 入力フォルダのバリデーション
        input_path = Path(parsed_args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"入力フォルダが存在しません: {parsed_args.input}")
        if not input_path.is_dir():
            raise NotADirectoryError(
                f"指定パスはフォルダではありません: {parsed_args.input}"
            )

        # 依存関係の遅延初期化
        if self._analyzer is None:
            # CLI引数を含めてAnalyzerConfigを一括生成
            analyzer_config = AnalyzerConfig.from_cli_args(
                result_max_workers=parsed_args.result_max_workers,
                max_dim=parsed_args.max_dim,
                max_memory_mb=parsed_args.max_memory_mb,
            )
            self._analyzer = ImageQualityAnalyzer(config=analyzer_config)

        if self._picker is None:
            seed = parsed_args.seed
            rng = random.Random(seed) if seed is not None else None

            # CLI引数を含めてSelectionConfigを一括生成
            selection_config = SelectionConfig.from_cli_args(
                batch_size=parsed_args.batch_size,
            )
            self._picker = GameScreenPicker(
                self._analyzer, config=selection_config, rng=rng
            )

        best, stats = self._picker.select(
            parsed_args.input,
            parsed_args.num,
            parsed_args.similarity,
            parsed_args.recursive,
        )

        FileUtils.copy_selected_items(best, parsed_args.output)

        ResultFormatter.display_results(best, stats)

    @staticmethod
    def validate_positive_int(value: str) -> int:
        """正の整数をバリデーションする.

        Args:
            value: コマンドラインから渡された文字列値

        Returns:
            バリデーション済みの整数値

        Raises:
            argparse.ArgumentTypeError: 値が正の整数でない場合
        """
        try:
            ivalue = int(value)
            if ivalue <= 0:
                raise argparse.ArgumentTypeError(
                    f"--num: 正の整数を指定してください（実際の値: {ivalue}）"
                )
            return ivalue
        except ValueError as e:
            raise argparse.ArgumentTypeError(f"'{value}' は整数ではありません") from e

    @staticmethod
    def validate_similarity_range(value: str) -> float:
        """類似度しきい値をバリデーションする（0.0 ~ 1.0）.

        Args:
            value: コマンドラインから渡された文字列値

        Returns:
            バリデーション済みの浮動小数点値

        Raises:
            argparse.ArgumentTypeError: 値が0.0~1.0の範囲外の場合
        """
        try:
            fvalue = float(value)
            if not 0.0 <= fvalue <= 1.0:
                raise argparse.ArgumentTypeError(
                    f"--similarity: 0.0~1.0の範囲で指定してください"
                    f"（実際の値: {fvalue}）"
                )
            return fvalue
        except ValueError as e:
            raise argparse.ArgumentTypeError(f"'{value}' は数値ではありません") from e

    @staticmethod
    def validate_positive_int_or_zero(value: str) -> int:
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
            argparse.ArgumentTypeError: 値が0以上の整数でない場合
        """
        try:
            ivalue = int(value)
            if ivalue < 0:
                raise argparse.ArgumentTypeError(
                    f"0以上の整数を指定してください（実際の値: {ivalue}）"
                )
            # ThreadPoolExecutor(max_workers=0) は ValueError になるため
            # 0 を指定した場合は 1 (シングルスレッド) に変換
            return ivalue if ivalue > 0 else 1
        except ValueError as e:
            raise argparse.ArgumentTypeError(f"'{value}' は整数ではありません") from e

    def _parse_arguments(self) -> argparse.Namespace:
        """コマンドライン引数をパースする.

        Returns:
            パースされた引数のNamespace
        """
        parser = argparse.ArgumentParser(description="Diverse Game Screen Picker")
        parser.add_argument("input", help="入力フォルダ")
        parser.add_argument("output", help="出力フォルダ")
        parser.add_argument(
            "-n", "--num", type=Main.validate_positive_int, default=10, help="選択枚数"
        )
        parser.add_argument(
            "-s",
            "--similarity",
            type=Main.validate_similarity_range,
            default=0.72,
            help="類似度しきい値(0.7~0.85推奨)",
        )
        parser.add_argument(
            "-r", "--recursive", action="store_true", help="サブフォルダも検索"
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=None,
            help="乱数シード（再現可能な結果を得るために指定）",
        )
        parser.add_argument(
            "--batch-size",
            type=Main.validate_positive_int,
            default=None,
            help=(
                "CLIP推論のバッチサイズ"
                "（デフォルト: 32、大きいほど高速だがメモリ消費増加）"
            ),
        )
        parser.add_argument(
            "--result-max-workers",
            type=Main.validate_positive_int_or_zero,
            default=None,
            help=(
                "結果構築の並列ワーカー数（デフォルト: 自動設定、0でシングルスレッド）"
            ),
        )
        parser.add_argument(
            "--max-dim",
            type=Main.validate_positive_int,
            default=720,
            help=(
                "画像リサイズ時の長辺の最大ピクセル数"
                "（デフォルト: 720、小さいほど高速だが精度低下）"
            ),
        )
        parser.add_argument(
            "--max-memory-mb",
            type=Main.validate_positive_int,
            default=512,
            help=(
                "チャンク処理時のメモリ予算（MB）"
                "（デフォルト: 512、大きいほどチャンクが大きくなる）"
            ),
        )
        return parser.parse_args(self.args)


def main() -> None:
    """CLIエントリポイント関数.

    pyproject.tomlの[project.scripts]から呼び出される。
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
        force=True,
    )
    Main().run()


if __name__ == "__main__":
    main()
