"""同一target acceptance suiteのprocess排他を所有する。"""

import errno
import fcntl
import os
import stat
from contextlib import suppress
from pathlib import Path
from threading import Lock
from types import TracebackType


class AcceptanceSuiteLock:
    """state回復より前から同一suiteの同時実行を拒否する。"""

    _process_guard = Lock()
    _process_paths: set[Path] = set()

    def __init__(self, path: Path, *, owned_root: Path) -> None:
        self._path = Path(os.path.abspath(path))
        self._owned_root = Path(os.path.abspath(owned_root))
        try:
            self._relative_path = self._path.relative_to(self._owned_root)
        except ValueError:
            raise ValueError(
                "Acceptance suite lockはartifact root外に作成できません"
            ) from None
        if not self._relative_path.parts:
            raise ValueError("Acceptance suite lock pathが不正です")
        self._descriptor: int | None = None

    def __enter__(self) -> "AcceptanceSuiteLock":
        """process内予約後にOSの非待機lockを取得する。"""
        with self._process_guard:
            if self._path in self._process_paths:
                raise self._already_running_error()
            self._process_paths.add(self._path)
        try:
            descriptor = self._open_lock_file()
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BaseException as error:
                os.close(descriptor)
                if isinstance(error, OSError) and error.errno in {
                    errno.EACCES,
                    errno.EAGAIN,
                }:
                    raise self._already_running_error() from error
                raise
            self._descriptor = descriptor
        except BaseException:
            self._release_process_path()
            raise
        return self

    def _open_lock_file(self) -> int:
        root_descriptor = self._open_owned_root()
        descriptor = root_descriptor
        try:
            for part in self._relative_path.parts[:-1]:
                descriptor = self._open_or_create_directory(descriptor, part)
            try:
                lock_descriptor = os.open(
                    self._relative_path.parts[-1],
                    os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC,
                    0o600,
                    dir_fd=descriptor,
                )
            except OSError:
                raise ValueError(
                    "Acceptance suite lock pathにsymbolic linkがあります"
                ) from None
            try:
                lock_mode = os.fstat(lock_descriptor).st_mode
            except BaseException:
                os.close(lock_descriptor)
                raise
            if not stat.S_ISREG(lock_mode):
                os.close(lock_descriptor)
                raise ValueError("Acceptance suite lockが通常fileではありません")
            return lock_descriptor
        finally:
            os.close(descriptor)

    def _open_owned_root(self) -> int:
        try:
            before = self._owned_root.lstat()
        except FileNotFoundError:
            raise ValueError("Acceptance artifact rootが存在しません") from None
        if stat.S_ISLNK(before.st_mode):
            raise ValueError("Acceptance suite lock pathにsymbolic linkがあります")
        if not stat.S_ISDIR(before.st_mode):
            raise ValueError("Acceptance artifact rootがdirectoryではありません")
        try:
            descriptor = os.open(
                self._owned_root,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            )
        except OSError:
            raise ValueError(
                "Acceptance suite lock pathにsymbolic linkがあります"
            ) from None
        try:
            matches_root = os.path.samestat(before, os.fstat(descriptor))
        except BaseException:
            os.close(descriptor)
            raise
        if not matches_root:
            os.close(descriptor)
            raise ValueError("Acceptance artifact rootが検証中に変更されました")
        return descriptor

    @staticmethod
    def _open_or_create_directory(parent_descriptor: int, name: str) -> int:
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        try:
            child_descriptor = os.open(name, flags, dir_fd=parent_descriptor)
        except FileNotFoundError:
            with suppress(FileExistsError):
                os.mkdir(name, mode=0o700, dir_fd=parent_descriptor)
            try:
                child_descriptor = os.open(name, flags, dir_fd=parent_descriptor)
            except OSError:
                raise ValueError(
                    "Acceptance suite lock pathにsymbolic linkがあります"
                ) from None
        except OSError:
            raise ValueError(
                "Acceptance suite lock pathにsymbolic linkがあります"
            ) from None
        os.close(parent_descriptor)
        return child_descriptor

    def __exit__(
        self,
        _exception_type: type[BaseException] | None,
        _exception: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        """終了経路にかかわらずOS lockとprocess内予約を解放する。"""
        descriptor = self._descriptor
        self._descriptor = None
        try:
            if descriptor is not None:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                finally:
                    os.close(descriptor)
        finally:
            self._release_process_path()

    def _already_running_error(self) -> ValueError:
        return ValueError(f"Acceptance suiteは実行中です: {self._path.stem}")

    def _release_process_path(self) -> None:
        with self._process_guard:
            self._process_paths.discard(self._path)
