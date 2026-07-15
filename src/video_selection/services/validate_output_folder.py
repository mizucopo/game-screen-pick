"""Output Folderを副作用前に検証する。"""

from pathlib import Path


def validate_output_folder(output_folder: Path) -> None:
    """walking skeletonが所有できる未作成pathであることを検証する。"""
    if output_folder.exists():
        msg = f"Output Folderは存在しない必要があります: {output_folder}"
        raise ValueError(msg)
