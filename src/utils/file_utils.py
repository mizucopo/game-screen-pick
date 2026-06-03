"""ファイル操作ユーティリティ."""

import logging
import shutil
from pathlib import Path

from ..models.output_record import OutputRecord
from ..services.output_planner import OutputPlanner

logger = logging.getLogger(__name__)


class FileUtils:
    """ファイル操作ユーティリティクラス.

    出力ディレクトリ内で一意なファイルパスを生成する機能と、
    選択されたアイテムをコピーする機能を提供する。
    """

    @staticmethod
    def get_unique_destination(dest_dir: Path, filename: str) -> Path:
        """出力ディレクトリ内で一意なファイルパスを生成する.

        同名のファイルが存在する場合は連番サフィックスを付与する。

        Args:
            dest_dir: 出力先ディレクトリのパス
            filename: 元のファイル名

        Returns:
            出力ディレクトリ内で一意なファイルパス
        """
        existing_filenames = (
            [path.name for path in dest_dir.iterdir()] if dest_dir.exists() else []
        )
        return dest_dir / OutputPlanner.get_unique_filename(
            filename,
            existing_filenames,
        )

    @staticmethod
    def build_renamed_filename(
        scene_name: str,
        index: int,
        suffix: str,
        requested_num: int,
    ) -> str:
        """scene名と連番から出力ファイル名を構築する.

        Args:
            scene_name: `play` / `event` の接頭辞
            index: sceneごとの連番（1始まり）
            suffix: 元ファイルの拡張子
            requested_num: CLIで要求された出力枚数

        Returns:
            リネーム済みのファイル名

        Raises:
            ValueError: requested_num が1未満の場合
        """
        return OutputPlanner.build_renamed_filename(
            scene_name=scene_name,
            index=index,
            suffix=suffix,
            requested_num=requested_num,
        )

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
            shutil.copy2(result.source_path, result.output_path)

    @staticmethod
    def copy_selected_items(
        output_record: OutputRecord,
        dest_dir: str,
        rename: bool = False,
        requested_num: int | None = None,
    ) -> OutputRecord:
        """選択されたアイテムを出力ディレクトリにコピーする.

        Args:
            output_record: 出力候補を含むrecord
            dest_dir: 出力先ディレクトリのパス
            rename: scene別の連番ファイル名で出力するかどうか
            requested_num: CLIで要求された出力枚数

        Returns:
            コピー先パスを反映した出力record
        """
        out = Path(dest_dir)
        out.mkdir(parents=True, exist_ok=True)
        planned_output_record = OutputPlanner.plan_selected_outputs(
            output_record,
            dest_dir,
            rename=rename,
            requested_num=requested_num,
            existing_filenames=[path.name for path in out.iterdir()],
        )
        FileUtils.copy_planned_outputs(planned_output_record)
        logger.info(f"{len(output_record.selected)} 件を {dest_dir} に保存しました。")
        return planned_output_record
