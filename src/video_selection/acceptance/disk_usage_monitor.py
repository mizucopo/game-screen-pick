"""acceptance phase中のcache/temp/staging容量sampler。"""

import os
from pathlib import Path
from threading import Event, Lock, Thread


class DiskUsageMonitor:
    """outputを除外してpersistent cacheとpeak追加容量を測る。"""

    def __init__(
        self,
        *,
        working_root: Path,
        output_parent: Path,
        cache_folder: Path,
        interval_seconds: float = 1.0,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("Disk sampler intervalは正の値が必要です")
        self._working_root = working_root
        self._output_parent = output_parent
        self._cache_folder = cache_folder
        self._interval_seconds = interval_seconds
        self._stop = Event()
        self._lock = Lock()
        self._peak_bytes = 0
        self._sample_count = 0
        self._thread: Thread | None = None

    def start(self) -> None:
        """現在容量をsampleしbackground samplerを開始する。"""
        if self._thread is not None:
            raise RuntimeError("Disk usage monitorは一度だけ開始できます")
        self._sample()
        self._thread = Thread(target=self._run, name="acceptance-disk-monitor")
        self._thread.daemon = True
        self._thread.start()

    def stop(self) -> dict[str, int]:
        """samplerを停止しpersistent cacheとpeak追加容量を返す。"""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self._interval_seconds * 4))
        self._sample()
        return {
            "persistent_cache_bytes": _tree_size(self._cache_folder),
            "peak_additional_bytes": self._peak_bytes,
            "disk_sample_count": self._sample_count,
        }

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            self._sample()

    def _sample(self) -> None:
        size = _tree_size(self._working_root) + _staging_size(self._output_parent)
        with self._lock:
            self._peak_bytes = max(self._peak_bytes, size)
            self._sample_count += 1


def _tree_size(root: Path) -> int:
    total = 0
    try:
        entries = tuple(os.scandir(root))
    except FileNotFoundError:
        return 0
    for entry in entries:
        try:
            if entry.is_symlink():
                total += entry.stat(follow_symlinks=False).st_size
            elif entry.is_dir(follow_symlinks=False):
                total += _tree_size(Path(entry.path))
            elif entry.is_file(follow_symlinks=False):
                total += entry.stat(follow_symlinks=False).st_size
        except FileNotFoundError:
            continue
    return total


def _staging_size(output_parent: Path) -> int:
    try:
        staging = tuple(
            path
            for path in output_parent.iterdir()
            if path.name.startswith(".") and path.name.endswith(".staging")
        )
    except FileNotFoundError:
        return 0
    return sum(_tree_size(path) for path in staging)
