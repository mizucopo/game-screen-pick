"""application実行 orchestration."""

import logging
from pathlib import Path

import click
import cv2

from ..analyzers.image_quality_analyzer import ImageQualityAnalyzer
from ..models.application_run_request import ApplicationRunRequest
from ..models.output_record import OutputRecord
from ..services.game_screen_picker import GameScreenPicker
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
        output_record = _select_output_record(request, input_path)
        output_record = _copy_selected_outputs(request, output_record)

        ResultFormatter.display_results(output_record)
        if request.report_json is not None:
            ReportWriter.write(request.report_json, output_record)

    except click.ClickException:
        raise
    except Exception as error:
        logger.error(f"予期しないエラーが発生しました: {type(error).__name__}: {error}")
        raise SystemExit(1) from error


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
        profile=request.profile,
        scene_mix=request.scene_mix,
        similarity=request.similarity,
        batch_size=request.batch_size,
        result_max_workers=request.result_max_workers,
        max_dim=request.max_dim,
        max_memory_gb=request.max_memory_gb,
    )

    with ImageQualityAnalyzer(config=analyzer_config) as analyzer:
        picker = GameScreenPicker(analyzer, config=selection_config)
        logger.info("画像処理を開始します...")

        selected, rejected, stats = picker.select(
            folder=str(input_path),
            num=request.num,
            recursive=request.recursive,
        )
        return OutputRecord.from_selection(selected, rejected, stats)


def _copy_selected_outputs(
    request: ApplicationRunRequest,
    output_record: OutputRecord,
) -> OutputRecord:
    """選択済みrecordを計画済み出力先へコピーする."""
    return FileUtils.copy_selected_items(
        output_record,
        request.output_dir,
        rename=request.rename,
        requested_num=request.num,
    )
