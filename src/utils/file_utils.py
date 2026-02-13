"""ファイル操作ユーティリティ."""

import logging
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class FileUtils:
    """ファイル操作ユーティリティクラス.

    出力ディレクトリ内で一意なファイルパスを生成する静的メソッドを提供する。
    """

    @staticmethod
    def get_unique_destination(dest_dir: Path, filename: str) -> Path:
        """出力ディレクトリ内で一意なファイルパスを生成する.

        同名のファイルが存在する場合、連番サフィックスを付与して衝突を回避する。
        ファイル名の拡張子は維持される。

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
    def copy_selected_items(selected: list[Any], dest_dir: str) -> None:
        """選択されたアイテムを出力ディレクトリにコピーする.

        Args:
            selected: path属性を持つオブジェクトのリスト
            dest_dir: 出力先ディレクトリのパス
        """
        out = Path(dest_dir)
        out.mkdir(parents=True, exist_ok=True)
        for res in selected:
            original_filename = Path(res.path).name
            unique_dest = FileUtils.get_unique_destination(out, original_filename)
            shutil.copy2(res.path, unique_dest)
        logger.info(f"{len(selected)} 件を {dest_dir} に保存しました。")
