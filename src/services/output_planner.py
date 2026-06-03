"""出力先計画を純粋に決定する."""

from collections import defaultdict
from collections.abc import Collection, Iterable
from pathlib import Path

from ..models.output_record import OutputRecord


class OutputPlanner:
    """選択候補の出力先パスを計画する."""

    @staticmethod
    def plan_selected_outputs(
        output_record: OutputRecord,
        dest_dir: str,
        rename: bool = False,
        requested_num: int | None = None,
        existing_filenames: Iterable[str] = (),
    ) -> OutputRecord:
        """選択候補の出力先パスをcopyなしで計画する.

        Args:
            output_record: 出力候補を含むrecord。
            dest_dir: 出力先ディレクトリのパス。
            rename: scene別の連番ファイル名で出力するかどうか。
            requested_num: CLIで要求された出力枚数。
            existing_filenames: 出力先に既に存在するファイル名。

        Returns:
            出力先パスを反映したrecord。

        Raises:
            ValueError: rename=True かつ requested_num が未指定の場合。
        """
        out = Path(dest_dir)
        rename_requested_num = OutputPlanner._resolve_rename_requested_num(
            rename,
            requested_num,
        )
        reserved_filenames = set(existing_filenames)
        scene_counters: dict[str, int] = defaultdict(int)
        planned_paths_by_source_path: dict[str, str] = {}

        for candidate in output_record.selected:
            if rename:
                scene_counters[candidate.scene_label] += 1
                filename = OutputPlanner.build_renamed_filename(
                    scene_name=candidate.scene_label,
                    index=scene_counters[candidate.scene_label],
                    suffix=candidate.suffix,
                    requested_num=rename_requested_num,
                )
            else:
                filename = candidate.filename

            unique_filename = OutputPlanner._get_unique_filename_from_reserved(
                filename,
                reserved_filenames,
            )
            reserved_filenames.add(unique_filename)
            planned_paths_by_source_path[candidate.source_path] = str(
                (out / unique_filename).resolve()
            )

        return output_record.with_selected_output_paths(planned_paths_by_source_path)

    @staticmethod
    def _resolve_rename_requested_num(
        rename: bool,
        requested_num: int | None,
    ) -> int:
        """rename時に必要な要求枚数を解決する."""
        if requested_num is None:
            if rename:
                msg = "rename=True の場合は requested_num の指定が必要です"
                raise ValueError(msg)
            return 0
        return requested_num

    @staticmethod
    def build_renamed_filename(
        scene_name: str,
        index: int,
        suffix: str,
        requested_num: int,
    ) -> str:
        """scene名と連番から出力ファイル名を構築する."""
        if requested_num < 1:
            msg = f"requested_numは正の整数である必要があります: {requested_num}"
            raise ValueError(msg)
        width = max(4, len(str(requested_num)))
        return f"{scene_name}{index:0{width}d}{suffix}"

    @staticmethod
    def get_unique_filename(
        filename: str,
        reserved_filenames: Iterable[str],
    ) -> str:
        """予約済みファイル名と衝突しないファイル名を生成する."""
        return OutputPlanner._get_unique_filename_from_reserved(
            filename,
            set(reserved_filenames),
        )

    @staticmethod
    def _get_unique_filename_from_reserved(
        filename: str,
        reserved_filenames: Collection[str],
    ) -> str:
        """予約済みファイル名collectionから衝突しないファイル名を生成する."""
        if filename not in reserved_filenames:
            return filename

        dest_path = Path(filename)
        stem = dest_path.stem
        suffix = dest_path.suffix

        counter = 1
        while True:
            new_filename = f"{stem}_{counter}{suffix}"
            if new_filename not in reserved_filenames:
                return new_filename
            counter += 1
