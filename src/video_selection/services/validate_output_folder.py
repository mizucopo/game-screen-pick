"""Output Folderを副作用前に検証する。"""

from pathlib import Path

from .validate_canonical_selection_report import (
    load_validated_canonical_selection_report,
)


def validate_output_folder(
    input_folder: Path,
    output_folder: Path,
    *,
    allow_completed_canonical_output: bool = False,
) -> None:
    """inputと分離された未作成、空、または検証済みOutputを検証する。"""
    normalized_input = input_folder.resolve(strict=False)
    normalized_output = output_folder.resolve(strict=False)
    if (
        normalized_input == normalized_output
        or normalized_input in normalized_output.parents
        or normalized_output in normalized_input.parents
    ):
        msg = "Video Input FolderとOutput Folderは相互の親子pathにできません"
        raise ValueError(msg)

    if output_folder.is_symlink():
        msg = f"Output Folderにsymlinkは指定できません: {output_folder}"
        raise ValueError(msg)
    _validate_output_parent(output_folder)
    if not output_folder.exists():
        return
    if not output_folder.is_dir():
        msg = f"Output Folderは存在しないか空である必要があります: {output_folder}"
        raise ValueError(msg)
    if not any(output_folder.iterdir()):
        return
    if allow_completed_canonical_output:
        try:
            load_validated_canonical_selection_report(output_folder)
        except ValueError:
            pass
        else:
            return
    msg = f"Output Folderは存在しないか空である必要があります: {output_folder}"
    raise ValueError(msg)


def _validate_output_parent(output_folder: Path) -> None:
    """最も近い既存の親componentがdirectoryであることを検証する。"""
    parent = output_folder.parent
    while not parent.exists():
        if parent.is_symlink():
            msg = f"Output Folderの親pathにdangling symlinkがあります: {parent}"
            raise ValueError(msg)
        next_parent = parent.parent
        if next_parent == parent:
            break
        parent = next_parent
    if not parent.is_dir():
        msg = f"Output Folderの親pathはdirectoryである必要があります: {parent}"
        raise ValueError(msg)
