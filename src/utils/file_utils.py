"""ファイル操作ユーティリティ."""

import logging
import shutil
from pathlib import Path

from ..models.output_record import OutputRecord
from ..services.output_planner import OutputPlanner

logger = logging.getLogger(__name__)


class FileUtils:
    """ファイル操作ユーティリティクラス.

    scene別連番の出力計画に従って、選択されたアイテムをコピーする機能を提供する。
    """

    @staticmethod
    def copy_planned_outputs(output_record: OutputRecord) -> None:
        """計画済みの出力パスへ選択候補をコピーする.

        Args:
            output_record: output_path が設定済みの出力record

        Raises:
            ValueError: 選択候補に output_path が設定されていない場合
        """
        for result in output_record.selected:
            if result.output_path is None:
                msg = "コピー対象の output_path が設定されていません"
                raise ValueError(msg)
            output_path = Path(result.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(result.source_path, output_path)

    @staticmethod
    def copy_selected_items(
        output_record: OutputRecord,
        dest_dir: str,
        requested_num: int | None = None,
    ) -> OutputRecord:
        """選択されたアイテムを出力ディレクトリにコピーする.

        Args:
            output_record: 出力候補を含むrecord
            dest_dir: 出力先ディレクトリのパス
            requested_num: CLIで要求された出力枚数

        Returns:
            コピー先パスを反映した出力record
        """
        out = Path(dest_dir)
        out.mkdir(parents=True, exist_ok=True)
        planned_output_record = OutputPlanner.plan_selected_outputs(
            output_record,
            dest_dir,
            requested_num=requested_num,
            existing_filenames=[path.name for path in out.iterdir()],
        )
        FileUtils.copy_planned_outputs(planned_output_record)
        logger.info(f"{len(output_record.selected)} 件を {dest_dir} に保存しました。")
        return planned_output_record
