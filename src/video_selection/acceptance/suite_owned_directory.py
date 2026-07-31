"""Acceptance suiteが所有するdirectory chainの安全性を検証する。"""

from pathlib import Path


def validate_suite_owned_directory_chain(
    suite_root: Path,
    *descendants: Path,
    suite_label: str,
) -> None:
    """suite rootから対象directoryまで外部symlinkを辿らないことを保証する。"""
    for path in (suite_root, *descendants):
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            raise ValueError(
                f"{suite_label} suite workが不正です。--reset-suiteが必要です"
            )
