"""application実行 orchestration."""

import logging
import sys
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

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    stream=sys.stdout,
    force=True,
)

logger = logging.getLogger(__name__)


def run_application(request: ApplicationRunRequest) -> None:
    """画像選定applicationを実行する."""
    if request.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    cv2.setNumThreads(1)

    try:
        input_path = Path(request.input_dir)
        if not input_path.is_dir():
            raise click.BadParameter(
                f"指定パスはフォルダではありません: {request.input_dir}",
                param_hint="input_dir",
            )

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
            output_record = OutputRecord.from_selection(selected, rejected, stats)
            output_record = FileUtils.copy_selected_items(
                output_record,
                request.output_dir,
                rename=request.rename,
                requested_num=request.num,
            )
            ResultFormatter.display_results(output_record)
            if request.report_json is not None:
                ReportWriter.write(
                    request.report_json,
                    output_record,
                )

    except click.ClickException:
        raise
    except Exception as error:
        logger.error(f"予期しないエラーが発生しました: {type(error).__name__}: {error}")
        raise SystemExit(1) from error
