"""suite所有rootを基準に削除対象を検証して削除する。"""

import os
import shutil
import stat
from pathlib import Path

_DIRECTORY_OPEN_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_RMTREE_AVOIDS_SYMLINK_ATTACKS = shutil.rmtree.avoids_symlink_attacks


class SuiteOwnedDeletionBoundary:
    """検証済みdirectory handleからだけsuite-owned artifactを削除する。"""

    def __init__(self, owned_root: Path) -> None:
        self._owned_root = Path(os.path.abspath(owned_root))

    def validate_directory(self, path: Path, label: str) -> None:
        """rootから対象までが通常directoryだけであることを検証する。"""
        self._validate(path, label, target_kind="directory")

    def validate_file(self, path: Path, label: str) -> None:
        """rootから対象fileまでに外部参照がないことを検証する。"""
        self._validate(path, label, target_kind="file")

    def remove_directory(self, path: Path, label: str) -> None:
        """開いた親directoryを基準にrecursive deletionを完了する。"""
        self.validate_directory(path, label)
        opened = self._open_parent(path, label)
        if opened is None:
            return
        parent_descriptor, target_name = opened
        try:
            mode = self._target_mode(parent_descriptor, target_name)
            if mode is None:
                return
            if stat.S_ISLNK(mode):
                raise ValueError(f"{label}の途中階層がsymbolic linkです")
            if not stat.S_ISDIR(mode):
                raise ValueError(f"{label}の途中階層が通常directoryではありません")
            if not _RMTREE_AVOIDS_SYMLINK_ATTACKS:
                raise ValueError(f"{label}を安全に削除できない実行環境です")
            shutil.rmtree(target_name, dir_fd=parent_descriptor)
            if self._target_mode(parent_descriptor, target_name) is not None:
                raise ValueError(f"{label}を完全に削除できません")
        except OSError:
            raise ValueError(f"{label}を完全に削除できません") from None
        finally:
            os.close(parent_descriptor)

    def remove_file(self, path: Path, label: str) -> None:
        """開いた親directoryを基準に通常fileだけを削除する。"""
        self.validate_file(path, label)
        opened = self._open_parent(path, label)
        if opened is None:
            return
        parent_descriptor, target_name = opened
        try:
            mode = self._target_mode(parent_descriptor, target_name)
            if mode is None:
                return
            if stat.S_ISLNK(mode):
                raise ValueError(f"{label}の途中階層がsymbolic linkです")
            if not stat.S_ISREG(mode):
                raise ValueError(f"{label}が通常fileではありません")
            os.unlink(target_name, dir_fd=parent_descriptor)
            if self._target_mode(parent_descriptor, target_name) is not None:
                raise ValueError(f"{label}を完全に削除できません")
        except OSError:
            raise ValueError(f"{label}を削除できません") from None
        finally:
            os.close(parent_descriptor)

    def _validate(self, path: Path, label: str, *, target_kind: str) -> None:
        target, relative = self._relative_target(path, label)
        current = self._owned_root
        chain = [self._owned_root]
        for part in relative.parts:
            current /= part
            chain.append(current)
        for candidate in chain:
            try:
                mode = candidate.lstat().st_mode
            except FileNotFoundError:
                return
            if stat.S_ISLNK(mode):
                raise ValueError(f"{label}の途中階層がsymbolic linkです")
            is_target = candidate == target
            if not is_target or target_kind == "directory":
                if not stat.S_ISDIR(mode):
                    raise ValueError(f"{label}の途中階層が通常directoryではありません")
            elif not stat.S_ISREG(mode):
                raise ValueError(f"{label}が通常fileではありません")

    def _open_parent(self, path: Path, label: str) -> tuple[int, str] | None:
        _target, relative = self._relative_target(path, label)
        root_descriptor = self._open_root(label)
        if root_descriptor is None:
            return None
        descriptor = root_descriptor
        try:
            for part in relative.parts[:-1]:
                try:
                    child_descriptor = os.open(
                        part,
                        _DIRECTORY_OPEN_FLAGS,
                        dir_fd=descriptor,
                    )
                except FileNotFoundError:
                    os.close(descriptor)
                    return None
                except OSError:
                    raise ValueError(
                        f"{label}の途中階層がsymbolic linkまたは非directoryです"
                    ) from None
                os.close(descriptor)
                descriptor = child_descriptor
            return descriptor, relative.parts[-1]
        except BaseException:
            os.close(descriptor)
            raise

    def _open_root(self, label: str) -> int | None:
        try:
            before = self._owned_root.lstat()
        except FileNotFoundError:
            return None
        if stat.S_ISLNK(before.st_mode):
            raise ValueError(f"{label}の途中階層がsymbolic linkです")
        if not stat.S_ISDIR(before.st_mode):
            raise ValueError(f"{label}の途中階層が通常directoryではありません")
        try:
            descriptor = os.open(self._owned_root, _DIRECTORY_OPEN_FLAGS)
        except OSError:
            raise ValueError(
                f"{label}の途中階層がsymbolic linkまたは非directoryです"
            ) from None
        try:
            after = os.fstat(descriptor)
        except BaseException:
            os.close(descriptor)
            raise
        if not os.path.samestat(before, after):
            os.close(descriptor)
            raise ValueError(f"{label}の途中階層が検証中に変更されました")
        return descriptor

    def _relative_target(self, path: Path, label: str) -> tuple[Path, Path]:
        target = Path(os.path.abspath(path))
        try:
            relative = target.relative_to(self._owned_root)
        except ValueError:
            raise ValueError(f"{label}はsuite所有directory外にあります") from None
        if not relative.parts:
            raise ValueError(f"{label}はsuite所有root自体を対象にできません")
        return target, relative

    @staticmethod
    def _target_mode(parent_descriptor: int, target_name: str) -> int | None:
        try:
            return os.stat(
                target_name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            ).st_mode
        except FileNotFoundError:
            return None
