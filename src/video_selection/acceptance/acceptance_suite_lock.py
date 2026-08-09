"""同一target acceptance suiteのprocess排他を所有する。"""

import errno
import fcntl
import os
from pathlib import Path
from threading import Lock
from types import TracebackType


class AcceptanceSuiteLock:
    """state回復より前から同一suiteの同時実行を拒否する。"""

    _process_guard = Lock()
    _process_paths: set[Path] = set()

    def __init__(self, path: Path) -> None:
        self._path = path.absolute()
        self._descriptor: int | None = None

    def __enter__(self) -> "AcceptanceSuiteLock":
        """process内予約後にOSの非待機lockを取得する。"""
        with self._process_guard:
            if self._path in self._process_paths:
                raise self._already_running_error()
            self._process_paths.add(self._path)
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(self._path, os.O_CREAT | os.O_RDWR, 0o600)
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
