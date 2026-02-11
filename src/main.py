"""game_screen_pick - 高精度・コンテンツ多様性重視選択ツール."""

import argparse
import shutil
from pathlib import Path
from typing import List

from .analyzers import ImageQualityAnalyzer
from .models.genre_weights import GenreWeights
from .models.image_metrics import ImageMetrics
from .services import GameScreenPicker
from .utils.file_utils import FileUtils


class Main:
    """CLIメインクラス.

    画像選択ツールのコマンドラインインターフェースを提供する。
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

        # 依存関係の遅延初期化（引数パース後にジャンルが決まるため）
        if self._analyzer is None:
            self._analyzer = ImageQualityAnalyzer(parsed_args.genre)
        if self._picker is None:
            self._picker = GameScreenPicker(self._analyzer)

        best = self._picker.select(
            parsed_args.input,
            parsed_args.num,
            parsed_args.similarity,
            parsed_args.recursive,
        )

        if parsed_args.copy_to and best:
            self._copy_selected_images(best, parsed_args.copy_to)

        self._display_results(best)

    def _validate_positive_int(self, value: str) -> int:
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

    def _validate_similarity_range(self, value: str) -> float:
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

    def _parse_arguments(self) -> argparse.Namespace:
        """コマンドライン引数をパースする.

        Returns:
            パースされた引数のNamespace
        """
        genre_choices = sorted(GenreWeights.DEFAULT_WEIGHTS.keys())

        parser = argparse.ArgumentParser(description="Diverse Game Screen Picker")
        parser.add_argument("input", help="入力フォルダ")
        parser.add_argument("-c", "--copy-to", help="出力フォルダ")
        parser.add_argument(
            "-n", "--num", type=self._validate_positive_int, default=10, help="選択枚数"
        )
        parser.add_argument(
            "-g",
            "--genre",
            default="mixed",
            choices=genre_choices,
            help="ジャンル (デフォルト: mixed)",
        )
        parser.add_argument(
            "-s",
            "--similarity",
            type=self._validate_similarity_range,
            default=0.82,
            help="類似度しきい値(0.7~0.85推奨)",
        )
        parser.add_argument(
            "-r", "--recursive", action="store_true", help="サブフォルダも検索"
        )
        return parser.parse_args(self.args)

    def _copy_selected_images(
        self, selected: List[ImageMetrics], dest_dir: str
    ) -> None:
        """選択された画像を出力ディレクトリにコピーする.

        Args:
            selected: 選択された画像メトリクスのリスト
            dest_dir: 出力先ディレクトリのパス
        """
        out = Path(dest_dir)
        out.mkdir(parents=True, exist_ok=True)
        for res in selected:
            original_filename = Path(res.path).name
            unique_dest = FileUtils.get_unique_destination(out, original_filename)
            shutil.copy2(res.path, unique_dest)
        print(f"\n{len(selected)} 枚を {dest_dir} に保存しました（多様性確保済み）。")

    def _display_results(self, selected: List[ImageMetrics]) -> None:
        """選択結果を表示する.

        Args:
            selected: 選択された画像メトリクスのリスト
        """
        print("\n--- 選択された画像一覧 ---")
        for i, res in enumerate(selected):
            print(f"[{i + 1}] {Path(res.path).name} (Score: {res.total_score:.2f})")


def main() -> None:
    """CLIエントリポイント関数.

    pyproject.tomlの[project.scripts]から呼び出される。
    """
    Main().run()


if __name__ == "__main__":
    main()
