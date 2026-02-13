"""game_screen_pick - 高精度・コンテンツ多様性重視選択ツール."""

import argparse
import random
from pathlib import Path

from .analyzers.image_quality_analyzer import ImageQualityAnalyzer
from .cache.feature_cache import FeatureCache
from .models.analyzer_config import AnalyzerConfig
from .models.selection_config import SelectionConfig
from .services.game_screen_picker import GameScreenPicker
from .utils.file_utils import FileUtils
from .utils.result_formatter import ResultFormatter


class Main:
    """CLIメインクラス.

    画像選択ツールのコマンドラインインターフェースを提供する。
    """

    # デフォルトのキャッシュディレクトリ
    DEFAULT_CACHE_DIR = Path.home() / ".cache" / "game-screen-pick"
    DEFAULT_CACHE_FILE = DEFAULT_CACHE_DIR / "cache.sqlite3"

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

        # キャッシュ設定
        cache = None
        if not parsed_args.no_cache:
            cache_path = parsed_args.cache_file
            if cache_path is None:
                if parsed_args.copy_to:
                    # --copy-to指定時は出力フォルダ直下にキャッシュを保存
                    cache_path = Path(parsed_args.copy_to) / "cache.sqlite3"
                else:
                    # デフォルト: ユーザーキャッシュディレクトリ
                    cache_path = Main.DEFAULT_CACHE_FILE
            if cache_path:
                cache = FeatureCache(cache_path)

        # 依存関係の遅延初期化
        if self._analyzer is None:
            # AnalyzerConfigにCLI引数を反映
            analyzer_config = AnalyzerConfig()
            if parsed_args.result_max_workers is not None:
                analyzer_config.result_max_workers = parsed_args.result_max_workers

            self._analyzer = ImageQualityAnalyzer(cache=cache, config=analyzer_config)

        if self._picker is None:
            seed = parsed_args.seed
            rng = random.Random(seed) if seed is not None else None

            # SelectionConfigにCLI引数を反映
            selection_config = SelectionConfig()
            if parsed_args.batch_size is not None:
                selection_config.batch_size = parsed_args.batch_size

            self._picker = GameScreenPicker(
                self._analyzer, config=selection_config, rng=rng
            )

        best, stats = self._picker.select(
            parsed_args.input,
            parsed_args.num,
            parsed_args.similarity,
            parsed_args.recursive,
        )

        if parsed_args.copy_to and best:
            FileUtils.copy_selected_items(best, parsed_args.copy_to)

        ResultFormatter.display_results(best, stats)

        # キャッシュを閉じる
        if cache:
            cache.close()

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

        Args:
            value: コマンドラインから渡された文字列値

        Returns:
            バリデーション済みの整数値

        Raises:
            argparse.ArgumentTypeError: 値が0以上の整数でない場合
        """
        try:
            ivalue = int(value)
            if ivalue < 0:
                raise argparse.ArgumentTypeError(
                    f"0以上の整数を指定してください（実際の値: {ivalue}）"
                )
            return ivalue
        except ValueError as e:
            raise argparse.ArgumentTypeError(f"'{value}' は整数ではありません") from e

    def _parse_arguments(self) -> argparse.Namespace:
        """コマンドライン引数をパースする.

        Returns:
            パースされた引数のNamespace
        """
        parser = argparse.ArgumentParser(description="Diverse Game Screen Picker")
        parser.add_argument("input", help="入力フォルダ")
        parser.add_argument("-c", "--copy-to", help="出力フォルダ")
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
            "--cache-file",
            type=str,
            default=None,
            help=(
                "キャッシュデータベースのパス"
                "（デフォルト: ~/.cache/game-screen-pick/cache.sqlite3、"
                "--copy-to指定時は {出力フォルダ}/cache.sqlite3）"
            ),
        )
        parser.add_argument(
            "--no-cache",
            action="store_true",
            help="キャッシュを無効化する（デフォルトでは有効）",
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
        return parser.parse_args(self.args)


def main() -> None:
    """CLIエントリポイント関数.

    pyproject.tomlの[project.scripts]から呼び出される。
    """
    Main().run()


if __name__ == "__main__":
    main()
