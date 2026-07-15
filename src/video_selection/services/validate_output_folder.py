"""Output Folderを副作用前に検証する。"""

from pathlib import Path


def validate_output_folder(input_folder: Path, output_folder: Path) -> None:
    """inputと分離された未作成または空のOutput Folderを検証する。"""
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
    if not output_folder.exists():
        return
    if not output_folder.is_dir() or any(output_folder.iterdir()):
        msg = f"Output Folderは存在しないか空である必要があります: {output_folder}"
        raise ValueError(msg)
    output_folder.rmdir()
