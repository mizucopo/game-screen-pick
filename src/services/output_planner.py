"""出力先計画を純粋に決定する."""

from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from ..models.output_record import OutputRecord


class OutputPlanner:
    """選択候補の出力先パスを計画する."""

    @staticmethod
    def plan_selected_outputs(
        output_record: OutputRecord,
        dest_dir: str,
        requested_num: int | None = None,
        existing_filenames: Iterable[str] = (),
    ) -> OutputRecord:
        """選択候補の出力先パスをcopyなしで計画する.

        Args:
            output_record: 出力候補を含むrecord。
            dest_dir: 出力先ディレクトリのパス。
            requested_num: CLIで要求された出力枚数。
            existing_filenames: 出力先に既に存在するファイル名。

        Returns:
            出力先パスを反映したrecord。

        Raises:
            ValueError: requested_num が未指定の場合。
        """
        if requested_num is None:
            msg = "scene別連番出力の場合は requested_num の指定が必要です"
            raise ValueError(msg)

        out = Path(dest_dir)
        reserved_collision_keys = OutputPlanner._build_reserved_collision_keys(
            existing_filenames
        )
        scene_counters: dict[str, int] = defaultdict(int)
        planned_paths_by_source_path: dict[str, str] = {}

        for candidate in output_record.selected:
            scene_slug = candidate.scene_slug
            scene_counters[scene_slug] += 1
            scene_index = scene_counters[scene_slug]
            filename = OutputPlanner.build_scene_numbered_filename(
                scene_name=scene_slug,
                index=scene_index,
                suffix=candidate.suffix,
                requested_num=requested_num,
            )
            unique_filename = OutputPlanner._get_unique_filename_from_collision_keys(
                filename,
                reserved_collision_keys,
            )
            reserved_collision_keys.add(
                OutputPlanner._build_collision_key(unique_filename)
            )
            planned_paths_by_source_path[candidate.source_path] = str(
                (out / unique_filename).resolve()
            )

        return output_record.with_selected_output_paths(planned_paths_by_source_path)

    @staticmethod
    def build_scene_numbered_filename(
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
    def _get_unique_filename_from_collision_keys(
        filename: str,
        reserved_collision_keys: set[str],
    ) -> str:
        """予約済み衝突keyから衝突しないファイル名を生成する."""
        if OutputPlanner._build_collision_key(filename) not in reserved_collision_keys:
            return filename

        dest_path = Path(filename)
        stem = dest_path.stem
        suffix = dest_path.suffix

        counter = 1
        while True:
            new_filename = f"{stem}_{counter}{suffix}"
            if (
                OutputPlanner._build_collision_key(new_filename)
                not in reserved_collision_keys
            ):
                return new_filename
            counter += 1

    @staticmethod
    def _build_reserved_collision_keys(filenames: Iterable[str]) -> set[str]:
        """予約済みファイル名を衝突判定用keyへ変換する."""
        return {OutputPlanner._build_collision_key(filename) for filename in filenames}

    @staticmethod
    def _build_collision_key(filename: str) -> str:
        """filesystem上の大小文字差による衝突を避けるkeyを生成する."""
        return filename.casefold()
