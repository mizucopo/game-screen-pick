"""Video Input Folder単位の非待機排他lock。"""

import fcntl
from pathlib import Path
from types import TracebackType
from typing import IO


class InputFolderLock:
    """同じVideo Input Folderの同時runを拒否するOS lock。"""

    def __init__(self, input_folder: Path) -> None:
        self._input_folder = input_folder
        self._lock_file: IO[bytes] | None = None

    @property
    def is_held(self) -> bool:
        """このinstanceがlockを保持しているか返す。"""
        return self._lock_file is not None

    @property
    def processing_cache_folder(self) -> Path:
        """このInput Lockが保護するprocessing cache rootを返す。"""
        return self._input_folder / ".game-screen-pick" / "cache"

    def __enter__(self) -> "InputFolderLock":
        """lock fileを開き非待機exclusive lockを取得する。"""
        metadata_folder = self._input_folder / ".game-screen-pick"
        if metadata_folder.is_symlink() or (
            metadata_folder.exists() and not metadata_folder.is_dir()
        ):
            msg = ".game-screen-pickには通常directoryが必要です"
            raise RuntimeError(msg)
        metadata_folder.mkdir(exist_ok=True)
        lock_file = (metadata_folder / "input.lock").open("a+b")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock_file.close()
            msg = "このVideo Input Folderは既に実行中です"
            raise RuntimeError(msg) from None
        self._lock_file = lock_file
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """保持中lockを必ず解放する。"""
        del exception_type, exception, traceback
        lock_file = self._lock_file
        if lock_file is None:
            return
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()
            self._lock_file = None

    def assert_held_for(self, cache_folder: Path) -> None:
        """指定cache rootに対応するlock保持を検証する。"""
        if not self.is_held or cache_folder != self.processing_cache_folder:
            msg = "processing cacheの変更には対応するInput Lockが必要です"
            raise RuntimeError(msg)
