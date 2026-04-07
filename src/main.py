"""game-screen-pick のCLIエントリポイント."""

import logging
import sys
from pathlib import Path

import click

from .analyzers.image_quality_analyzer import ImageQualityAnalyzer
from .models.analyzer_config import AnalyzerConfig
from .models.scene_mix import SceneMix
from .models.selection_config import SelectionConfig
from .services.game_screen_picker import GameScreenPicker
from .utils.config_loader import ConfigLoader
from .utils.file_utils import FileUtils
from .utils.report_writer import ReportWriter
from .utils.result_formatter import ResultFormatter

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    stream=sys.stdout,
    force=True,
)

logger = logging.getLogger(__name__)


class Main:
    """CLIメインクラス.

    入力値の検証、設定ファイルとCLI引数の解決、
    Analyzer / Picker の初期化、結果出力までの実行フローをまとめる。
    """

    def __init__(self, args: list[str]):
        """Mainクラスを初期化する.

        Args:
            args: コマンドライン引数。テスト時は空リストを渡す。
        """
        self.args = args

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def validate_similarity_range(value: str | None) -> float | None:
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

    @staticmethod
    def parse_scene_mix(value: str | None) -> SceneMix | None:
        """scene mix文字列を `SceneMix` へ変換する.

        受け付ける形式は `play=0.7,event=0.3` のみで、
        2要素が揃っていることを前提とする。合計値の検証は
        `SceneMix` モデルのバリデーションへ委ねる。

        Args:
            value: CLIで指定されたscene mix文字列。

        Returns:
            解析済みの `SceneMix` 。未指定時は `None` を返す。

        Raises:
            click.BadParameter: 形式不正、要素不足、数値変換失敗、整合性検証失敗時。
        """
        if value is None:
            return None

        pairs = {}
        for item in value.split(","):
            key, separator, raw_score = item.strip().partition("=")
            if separator != "=" or not key:
                raise click.BadParameter(
                    "scene-mixは play=0.7,event=0.3 形式で指定してください"
                )
            try:
                pairs[key] = float(raw_score)
            except ValueError as error:
                raise click.BadParameter(
                    f"scene-mixの値は数値である必要があります: {item}"
                ) from error

        expected_keys = {"play", "event"}
        if set(pairs) != expected_keys:
            raise click.BadParameter("scene-mixには play,event の2要素が必要です")
        try:
            return SceneMix(
                play=pairs["play"],
                event=pairs["event"],
            )
        except ValueError as error:
            raise click.BadParameter(str(error)) from error

    @staticmethod
    def _resolve_configs(
        config_path: str | None,
        profile: str | None,
        scene_mix: SceneMix | None,
        similarity: float | None,
        batch_size: int | None,
        result_max_workers: int | None,
        max_dim: int,
        max_memory_gb: int,
    ) -> tuple[AnalyzerConfig, SelectionConfig]:
        """解析設定と選択設定を構築する."""
        analyzer_config = AnalyzerConfig.from_cli_args(
            result_max_workers=result_max_workers,
            max_dim=max_dim,
            max_memory_gb=max_memory_gb,
        )
        selection_config = Main.build_selection_config(
            config_path=config_path,
            profile=profile,
            scene_mix=scene_mix,
            similarity=similarity,
            batch_size=batch_size,
        )
        return analyzer_config, selection_config

    @staticmethod
    def build_selection_config(
        *,
        config_path: str | None,
        profile: str | None,
        scene_mix: SceneMix | None,
        similarity: float | None,
        batch_size: int | None,
    ) -> SelectionConfig:
        """設定ファイルとCLI引数から `SelectionConfig` を作成する.

        優先順位は `CLI override > config file > built-in default` とする。
        このメソッドは設定ファイルを部分的な辞書へ変換したうえで、
        CLIで明示指定された値だけを上書きし、最終的な設定モデルを組み立てる。

        Args:
            config_path: TOML設定ファイルへのパス。
            profile: CLIから明示指定された実行プロファイル。
            scene_mix: CLIから明示指定されたscene mix比率。
            similarity: CLIから明示指定された類似度しきい値。
            batch_size: CLIから明示指定されたCLIPバッチサイズ。

        Returns:
            実行時にそのまま使える `SelectionConfig` 。
        """
        config_values = ConfigLoader.load(config_path)
        cli_overrides = {
            "profile": profile,
            "scene_mix": scene_mix,
            "similarity_threshold": similarity,
            "batch_size": batch_size,
        }
        merged = {
            **config_values,
            **{key: value for key, value in cli_overrides.items() if value is not None},
        }
        return SelectionConfig.from_cli_args(**merged)

    @click.command()
    @click.option(
        "-n",
        "--num",
        default=100,
        type=int,
        callback=lambda _ctx, _param, x: Main.validate_positive_int(x),
        help="選択枚数",
    )
    @click.option(
        "-s",
        "--similarity",
        default=None,
        type=float,
        callback=lambda _ctx, _param, x: Main.validate_similarity_range(x),
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
        "--scene-mix",
        callback=lambda _ctx, _param, x: Main.parse_scene_mix(x),
        default=None,
        help="画面種別比率。例: play=0.7,event=0.3",
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
        callback=lambda _ctx, _param, x: Main.validate_positive_int(x),
        default=None,
        help="CLIP推論のバッチサイズ",
    )
    @click.option(
        "--result-max-workers",
        type=int,
        callback=lambda _ctx, _param, x: Main.validate_positive_int_or_zero(x),
        default=None,
        help="結果構築の並列ワーカー数（0でシングルスレッド）",
    )
    @click.option(
        "--max-dim",
        type=int,
        callback=lambda _ctx, _param, x: Main.validate_positive_int(x),
        default=720,
        help="画像リサイズ時の長辺の最大ピクセル数",
    )
    @click.option(
        "--max-memory-gb",
        type=int,
        callback=lambda _ctx, _param, x: Main.validate_positive_int(x),
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
    def _execute(
        num: int,
        similarity: float | None,
        recursive: bool,
        profile: str | None,
        config_path: str | None,
        scene_mix: SceneMix | None,
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

        入力パスの検証、Analyzer / Picker の初期化、画像選定、
        出力フォルダへのコピー、標準出力への集計表示、
        必要に応じたJSONレポート出力までを一括で行う。
        選択設定は `build_selection_config` を経由して解決されるため、
        優先順位は常に `CLI > config file > built-in default` になる。

        \b
        使用例:
          game-screen-pick -n 15 ./screenshots ./output
          game-screen-pick --rename ./screenshots ./output
          game-screen-pick --scene-mix play=0.7,event=0.3 ./in ./out

        Args:
            num: 選択枚数。
            similarity: 類似度しきい値。未指定時は設定ファイルまたは既定値を使う。
            recursive: サブフォルダを再帰的に探索するかどうか。
            profile: 選定プロファイル。 `auto` / `active` / `static` 。
            config_path: TOML設定ファイルのパス。
            scene_mix: CLIから上書きする画面種別比率。
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
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)

        try:
            input_path = Path(input_dir)
            if not input_path.is_dir():
                raise click.BadParameter(
                    f"指定パスはフォルダではありません: {input_dir}",
                    param_hint="input_dir",
                )

            analyzer_config, selection_config = Main._resolve_configs(
                config_path=config_path,
                profile=profile,
                scene_mix=scene_mix,
                similarity=similarity,
                batch_size=batch_size,
                result_max_workers=result_max_workers,
                max_dim=max_dim,
                max_memory_gb=max_memory_gb,
            )

            with ImageQualityAnalyzer(config=analyzer_config) as analyzer:
                picker = GameScreenPicker(analyzer, config=selection_config)
                logger.info("画像処理を開始します...")

                selected, rejected, stats = picker.select(
                    folder=str(input_path),
                    num=num,
                    recursive=recursive,
                )
                copied_paths_by_path = FileUtils.copy_selected_items(
                    selected,
                    output_dir,
                    rename=rename,
                    requested_num=num,
                )
                ResultFormatter.display_results(selected, stats)
                if report_json is not None:
                    ReportWriter.write(
                        report_json,
                        selected,
                        rejected,
                        stats,
                        output_paths_by_candidate_id=copied_paths_by_path,
                    )

        except click.ClickException:
            raise
        except Exception as error:
            logger.error(
                f"予期しないエラーが発生しました: {type(error).__name__}: {error}"
            )
            raise SystemExit(1) from error

    def run(self) -> None:
        """CLIを実行する.

        `self.args` を `sys.argv` へ一時反映し、Clickフローで `_execute` を呼び出す。

        Returns:
            なし。

        Raises:
            SystemExit: 実行時エラーを終了コードへ変換した場合。
        """
        original_argv = sys.argv
        try:
            sys.argv = ["game-screen-pick"] + self.args
            self._execute(standalone_mode=False)
        except click.ClickException as error:
            error.show()
            raise SystemExit(error.exit_code) from error
        finally:
            sys.argv = original_argv


def cli_main() -> None:
    """CLIエントリポイント関数.

    `pyproject.toml` の script entrypoint から呼ばれる薄いラッパーで、
    `Main` を生成して実行するだけに責務を限定する。
    """
    Main(sys.argv[1:]).run()


if __name__ == "__main__":
    cli_main()
