"""ファイル操作ユーティリティ."""

from pathlib import Path


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
