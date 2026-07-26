"""Video Scan並列制御用のOS・NVIDIA resource sampleを取得する。"""

import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from threading import Lock

from ..models.video_scan_resource_sample import VideoScanResourceSample

GpuQuery = Callable[[], str | None]
ProcReader = Callable[[str], str | None]
Clock = Callable[[], float]
LogicalCpuCount = Callable[[], int | None]
LoadAverage = Callable[[], tuple[float, float, float] | None]
CpuCounters = tuple[int, int]
CpuSnapshot = tuple[CpuCounters, dict[str, CpuCounters]]

_PARTITION_PATTERNS = (
    re.compile(r"sd[a-z]+\d+"),
    re.compile(r"vd[a-z]+\d+"),
    re.compile(r"xvd[a-z]+\d+"),
    re.compile(r"nvme\d+n\d+p\d+"),
    re.compile(r"mmcblk\d+p\d+"),
)
_IGNORED_DISK_PREFIXES = ("loop", "ram", "zram", "fd", "sr", "dm-")


class VideoScanResourceSampler:
    """pathやdevice名を公開せずrolling utilizationを取得する。"""

    def __init__(
        self,
        *,
        gpu_query: GpuQuery | None = None,
        proc_reader: ProcReader | None = None,
        clock: Clock | None = None,
        logical_cpu_count: LogicalCpuCount | None = None,
        load_average: LoadAverage | None = None,
    ) -> None:
        self._gpu_query = gpu_query or _query_nvidia_smi
        self._proc_reader = proc_reader or _read_proc
        self._clock = clock or time.monotonic
        self._logical_cpu_count = logical_cpu_count or os.cpu_count
        self._load_average = load_average or _load_average
        self._lock = Lock()
        self._previous_cpu: CpuSnapshot | None = None
        self._previous_disks: (
            tuple[
                float,
                dict[str, tuple[int, int, int, int]],
            ]
            | None
        ) = None

    def sample(self) -> VideoScanResourceSample | None:
        """取得可能なresource割合とdisk throughputを一つ返す。"""
        with self._lock:
            return self._sample_locked()

    def _sample_locked(self) -> VideoScanResourceSample | None:
        """procfs counter更新とNVIDIA queryを一度に直列化する。"""
        sampled_at = self._clock()
        cpu_percent, saturated_cores = self._cpu_metrics(self._safe_read("/proc/stat"))
        memory_percent = _memory_percent(self._safe_read("/proc/meminfo"))
        disk_busy, disk_read, disk_latency = self._disk_metrics(
            self._safe_read("/proc/diskstats"),
            sampled_at,
        )
        decoder, gpu, vram = _gpu_metrics(self._safe_gpu_query())
        values = (
            cpu_percent,
            memory_percent,
            decoder,
            gpu,
            vram,
            disk_busy,
            disk_read,
            disk_latency,
            saturated_cores,
        )
        if all(value is None for value in values):
            return None
        return VideoScanResourceSample(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            decoder_percent=decoder,
            gpu_percent=gpu,
            vram_percent=vram,
            disk_busy_percent=disk_busy,
            disk_read_mib_per_second=disk_read,
            disk_read_latency_ms=disk_latency,
            cpu_saturated_core_percent=saturated_cores,
        )

    def _cpu_metrics(
        self,
        proc_stat: str | None,
    ) -> tuple[float | None, float | None]:
        current = _cpu_snapshot(proc_stat)
        if current is None:
            return (self._load_average_percent(), None)
        previous = self._previous_cpu
        self._previous_cpu = current
        if previous is None:
            return (self._load_average_percent(), None)
        cpu_percent = _cpu_utilization(previous[0], current[0])
        core_utilizations = tuple(
            utilization
            for name, counters in current[1].items()
            if (old := previous[1].get(name)) is not None
            and (utilization := _cpu_utilization(old, counters)) is not None
        )
        saturated_percent = (
            sum(value >= 90.0 for value in core_utilizations)
            / len(core_utilizations)
            * 100
            if core_utilizations
            else None
        )
        return (cpu_percent, saturated_percent)

    def _load_average_percent(self) -> float | None:
        try:
            load_average = self._load_average()
            logical_cpus = self._logical_cpu_count()
        except (OSError, ValueError):
            return None
        if load_average is None or logical_cpus is None or logical_cpus < 1:
            return None
        return _bounded_percent(load_average[0] / logical_cpus * 100)

    def _disk_metrics(
        self,
        diskstats: str | None,
        sampled_at: float,
    ) -> tuple[float | None, float | None, float | None]:
        current = _disk_counters(diskstats)
        previous = self._previous_disks
        self._previous_disks = (sampled_at, current)
        if previous is None or not current:
            return (None, None, None)
        elapsed = sampled_at - previous[0]
        if elapsed <= 0:
            return (None, None, None)
        previous_counters = previous[1]
        read_sectors = 0
        reads_completed = 0
        read_latency_ms = 0
        busiest_percent = 0.0
        comparable = False
        for name, (reads, sectors, read_ms, busy_ms) in current.items():
            old = previous_counters.get(name)
            if (
                old is None
                or reads < old[0]
                or sectors < old[1]
                or read_ms < old[2]
                or busy_ms < old[3]
            ):
                continue
            comparable = True
            reads_completed += reads - old[0]
            read_sectors += sectors - old[1]
            read_latency_ms += read_ms - old[2]
            busiest_percent = max(
                busiest_percent,
                (busy_ms - old[3]) / (elapsed * 1000) * 100,
            )
        if not comparable:
            return (None, None, None)
        read_mib_per_second = read_sectors * 512 / 1024 / 1024 / elapsed
        return (
            _bounded_percent(busiest_percent),
            max(0.0, read_mib_per_second),
            (read_latency_ms / reads_completed if reads_completed > 0 else None),
        )

    def _safe_read(self, path: str) -> str | None:
        try:
            return self._proc_reader(path)
        except (OSError, ValueError):
            return None

    def _safe_gpu_query(self) -> str | None:
        try:
            return self._gpu_query()
        except (OSError, subprocess.SubprocessError, ValueError):
            return None


def _query_nvidia_smi() -> str | None:
    executable = shutil.which("nvidia-smi")
    wsl_executable = Path("/usr/lib/wsl/lib/nvidia-smi")
    if executable is None and wsl_executable.is_file():
        executable = str(wsl_executable)
    if executable is None:
        return None
    try:
        completed = subprocess.run(
            (
                executable,
                "--query-gpu=utilization.decoder,utilization.gpu,"
                "memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ),
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout


def _read_proc(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        return None


def _load_average() -> tuple[float, float, float] | None:
    try:
        return os.getloadavg()
    except OSError:
        return None


def _cpu_snapshot(proc_stat: str | None) -> CpuSnapshot | None:
    if proc_stat is None:
        return None
    snapshots: dict[str, CpuCounters] = {}
    for line in proc_stat.splitlines():
        fields = line.split()
        if not fields or (fields[0] != "cpu" and not fields[0].startswith("cpu")):
            continue
        totals = _cpu_counter_totals(fields[1:])
        if totals is not None:
            snapshots[fields[0]] = totals
    aggregate = snapshots.pop("cpu", None)
    if aggregate is None:
        return None
    return (aggregate, snapshots)


def _cpu_counter_totals(values: list[str]) -> CpuCounters | None:
    try:
        counters = tuple(int(value) for value in values)
    except ValueError:
        return None
    if len(counters) < 4:
        return None
    total = sum(counters)
    idle = counters[3] + (counters[4] if len(counters) > 4 else 0)
    return (total, idle)


def _cpu_utilization(
    previous: CpuCounters,
    current: CpuCounters,
) -> float | None:
    total_delta = current[0] - previous[0]
    idle_delta = current[1] - previous[1]
    if total_delta <= 0 or idle_delta < 0:
        return None
    return _bounded_percent((total_delta - idle_delta) / total_delta * 100)


def _memory_percent(meminfo: str | None) -> float | None:
    if meminfo is None:
        return None
    values: dict[str, int] = {}
    for line in meminfo.splitlines():
        key, separator, remainder = line.partition(":")
        if not separator:
            continue
        try:
            values[key] = int(remainder.strip().split()[0])
        except (IndexError, ValueError):
            continue
    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    if total is None or available is None or total <= 0:
        return None
    return _bounded_percent((total - available) / total * 100)


def _disk_counters(
    diskstats: str | None,
) -> dict[str, tuple[int, int, int, int]]:
    if diskstats is None:
        return {}
    counters: dict[str, tuple[int, int, int, int]] = {}
    for line in diskstats.splitlines():
        fields = line.split()
        if len(fields) < 13:
            continue
        name = fields[2]
        if _ignore_disk(name):
            continue
        try:
            counters[name] = (
                int(fields[3]),
                int(fields[5]),
                int(fields[6]),
                int(fields[12]),
            )
        except ValueError:
            continue
    return counters


def _ignore_disk(name: str) -> bool:
    return name.startswith(_IGNORED_DISK_PREFIXES) or any(
        pattern.fullmatch(name) is not None for pattern in _PARTITION_PATTERNS
    )


def _gpu_metrics(
    output: str | None,
) -> tuple[float | None, float | None, float | None]:
    if output is None:
        return (None, None, None)
    for line in output.splitlines():
        fields = tuple(item.strip() for item in line.split(","))
        if len(fields) != 4:
            continue
        try:
            decoder, gpu, memory_used, memory_total = (float(value) for value in fields)
        except ValueError:
            continue
        if memory_total <= 0:
            return (None, None, None)
        return (
            _bounded_percent(decoder),
            _bounded_percent(gpu),
            _bounded_percent(memory_used / memory_total * 100),
        )
    return (None, None, None)


def _bounded_percent(value: float) -> float:
    return min(100.0, max(0.0, value))
