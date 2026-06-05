"""application実行 orchestration."""

import logging
import shutil
from pathlib import Path

import click
import cv2

from ..analyzers.image_quality_analyzer import ImageQualityAnalyzer
from ..models.application_run_request import ApplicationRunRequest
from ..models.output_record import OutputRecord
from ..services.game_screen_picker import GameScreenPicker
from ..services.ollama_scene_analyzer import OllamaSceneAnalyzer
from ..utils.config_resolver import ConfigResolver
from ..utils.file_utils import FileUtils
from ..utils.report_writer import ReportWriter
from ..utils.result_formatter import ResultFormatter

logger = logging.getLogger(__name__)


def run_application(request: ApplicationRunRequest) -> None:
    """画像選定applicationを実行する."""
    if request.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    cv2.setNumThreads(1)

    try:
        input_path = _resolve_input_path(request.input_dir)
        _reset_cache_if_requested(request, input_path)
        output_record = _select_output_record(request, input_path)
        output_record = FileUtils.copy_selected_items(
            output_record,
            request.output_dir,
            rename=request.rename,
            requested_num=request.num,
        )

        ResultFormatter.display_results(output_record)
        if request.report_json is not None:
            ReportWriter.write(request.report_json, output_record)

    except click.ClickException:
        raise
    except KeyboardInterrupt as error:
        _log_keyboard_interrupt()
        raise SystemExit(130) from error
    except Exception as error:
        logger.error(f"予期しないエラーが発生しました: {type(error).__name__}: {error}")
        raise SystemExit(1) from error


def _log_keyboard_interrupt() -> None:
    """Ctrl+C中断時の案内を出力する."""
    logger.info("中断されました。再実行するとcacheから再開します。")


def _reset_cache_if_requested(
    request: ApplicationRunRequest,
    input_path: Path,
) -> None:
    """指定されている場合は入力フォルダ配下のcacheを削除する."""
    if not request.reset_cache:
        return
    game_screen_pick_dirs = input_path.rglob(".game-screen-pick")
    for cache_dir in (path / "cache" for path in game_screen_pick_dirs):
        if cache_dir.exists():
            shutil.rmtree(cache_dir)


def _resolve_input_path(input_dir: str) -> Path:
    """入力ディレクトリを検証してPathへ変換する."""
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise click.BadParameter(
            f"指定パスはフォルダではありません: {input_dir}",
            param_hint="input_dir",
        )
    return input_path


def _select_output_record(
    request: ApplicationRunRequest,
    input_path: Path,
) -> OutputRecord:
    """画像選定を実行して出力recordへ変換する."""
    analyzer_config, selection_config = ConfigResolver.resolve_configs(
        config_path=request.config_path,
        similarity=request.similarity,
        batch_size=request.batch_size,
        result_max_workers=request.result_max_workers,
        max_dim=request.max_dim,
        max_memory_gb=request.max_memory_gb,
        ollama_model=request.ollama_model,
        ollama_host=request.ollama_host,
        ollama_timeout=request.ollama_timeout,
        ollama_max_workers=request.ollama_max_workers,
        scene_hint=request.scene_hint,
    )

    with ImageQualityAnalyzer(config=analyzer_config) as analyzer:
        if selection_config.ollama is None:
            msg = "Ollama設定が解決されていません"
            raise ValueError(msg)
        scene_analyzer = OllamaSceneAnalyzer(selection_config.ollama)
        picker = GameScreenPicker(
            analyzer,
            scene_analyzer=scene_analyzer,
            config=selection_config,
        )
        logger.info("画像処理を開始します...")

        selected, rejected, stats = picker.select(
            folder=str(input_path),
            num=request.num,
            recursive=request.recursive,
        )
        return OutputRecord.from_selection(selected, rejected, stats)
