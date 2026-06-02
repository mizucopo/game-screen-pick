"""ファイル操作ユーティリティ."""

import logging
import shutil
from collections import defaultdict
from pathlib import Path

from ..models.output_record import OutputRecord

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
        dest_path = dest_dir / filename

        if not dest_path.exists():
            return dest_path

        stem = dest_path.stem
        suffix = dest_path.suffix

        counter = 1
        while True:
            new_filename = f"{stem}_{counter}{suffix}"
            new_path = dest_dir / new_filename
            if not new_path.exists():
                return new_path
            counter += 1

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
        if requested_num < 1:
            msg = f"requested_numは正の整数である必要があります: {requested_num}"
            raise ValueError(msg)
        width = max(4, len(str(requested_num)))
        return f"{scene_name}{index:0{width}d}{suffix}"

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
        scene_counters: dict[str, int] = defaultdict(int)
        copied_paths_by_path: dict[str, str] = {}
        for result in output_record.selected:
            if rename:
                if requested_num is None:
                    msg = "rename=True の場合は requested_num の指定が必要です"
                    raise ValueError(msg)
                scene_name = result.scene_label
                scene_counters[scene_name] += 1
                filename = FileUtils.build_renamed_filename(
                    scene_name=scene_name,
                    index=scene_counters[scene_name],
                    suffix=result.suffix,
                    requested_num=requested_num,
                )
            else:
                filename = result.filename
            unique_dest = FileUtils.get_unique_destination(out, filename)
            shutil.copy2(result.source_path, unique_dest)
            copied_paths_by_path[result.source_path] = str(unique_dest.resolve())
        logger.info(f"{len(output_record.selected)} 件を {dest_dir} に保存しました。")
        return output_record.with_selected_output_paths(copied_paths_by_path)
