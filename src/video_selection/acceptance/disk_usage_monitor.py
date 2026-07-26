"""acceptance phase中のcache/temp/staging容量sampler。"""

import os
from collections.abc import Callable
from pathlib import Path
from threading import Event, Lock, Thread

SizeProbe = Callable[[Path], int]


class DiskUsageMonitor:
    """persistent cacheとoutputを除外してpeak追加容量を測る。"""

    def __init__(
        self,
        *,
        working_root: Path,
        output_parent: Path,
        cache_folder: Path,
        interval_seconds: float = 1.0,
        join_timeout_seconds: float = 4.0,
        tree_size_probe: SizeProbe | None = None,
        staging_size_probe: SizeProbe | None = None,
    ) -> None:
        if interval_seconds <= 0 or join_timeout_seconds <= 0:
            raise ValueError("Disk sampler interval/停止timeoutは正の値が必要です")
        self._working_root = working_root
        self._output_parent = output_parent
        self._cache_folder = cache_folder
        self._interval_seconds = interval_seconds
        self._join_timeout_seconds = join_timeout_seconds
        self._tree_size = tree_size_probe or _tree_size
        self._uses_default_tree_size = tree_size_probe is None
        self._staging_size = staging_size_probe or _staging_size
        self._stop = Event()
        self._lock = Lock()
        self._peak_bytes = 0
        self._sample_count = 0
        self._sample_errors = 0
        self._thread: Thread | None = None

    def start(self) -> None:
        """現在容量をsampleしbackground samplerを開始する。"""
        if self._thread is not None:
            raise RuntimeError("Disk usage monitorは一度だけ開始できます")
        self._sample()
        self._thread = Thread(target=self._run, name="acceptance-disk-monitor")
        self._thread.daemon = True
        self._thread.start()

    def stop(self) -> dict[str, int | bool]:
        """samplerを停止しpersistent cacheとpeak追加容量を返す。"""
        self._stop.set()
        sampler_stopped = False
        if self._thread is not None:
            self._thread.join(timeout=self._join_timeout_seconds)
            sampler_stopped = not self._thread.is_alive()
        if sampler_stopped:
            self._sample()
        with self._lock:
            peak_bytes = self._peak_bytes
            sample_count = self._sample_count
            sample_errors = self._sample_errors
        return {
            "disk_sampling_complete": (
                sampler_stopped and sample_count > 0 and sample_errors == 0
            ),
            "persistent_cache_bytes": self._tree_size(self._cache_folder),
            "peak_additional_bytes": peak_bytes,
            "disk_sample_count": sample_count,
            "disk_sample_error_count": sample_errors,
        }

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            self._sample()

    def _sample(self) -> None:
        try:
            working_size = (
                _tree_size(
                    self._working_root,
                    excluded_root=self._cache_folder,
                )
                if self._uses_default_tree_size
                else max(
                    self._tree_size(self._working_root)
                    - self._tree_size(self._cache_folder),
                    0,
                )
            )
            size = working_size + self._staging_size(self._output_parent)
        except Exception:
            with self._lock:
                self._sample_errors += 1
            return
        with self._lock:
            self._peak_bytes = max(self._peak_bytes, size)
            self._sample_count += 1


def _tree_size(
    root: Path,
    *,
    excluded_root: Path | None = None,
) -> int:
    if excluded_root is not None and root == excluded_root:
        return 0
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
                total += _tree_size(
                    Path(entry.path),
                    excluded_root=excluded_root,
                )
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
