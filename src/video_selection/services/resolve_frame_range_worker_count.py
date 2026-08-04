"""Frame Range decodeのbounded worker数を解決する。"""

_MAX_FRAME_RANGE_WORKERS = 4
_LOGICAL_CPUS_PER_FRAME_RANGE_WORKER = 4


def resolve_frame_range_worker_count(
    range_count: int,
    *,
    logical_cpu_count: int,
) -> int:
    """range件数とlogical CPU容量から安全なworker数を返す。"""
    if range_count < 1:
        raise ValueError("Frame Range件数は正の整数である必要があります")
    if logical_cpu_count < 1:
        raise ValueError("logical CPU数は正の整数である必要があります")
    cpu_workers = max(
        1,
        logical_cpu_count // _LOGICAL_CPUS_PER_FRAME_RANGE_WORKER,
    )
    return min(range_count, _MAX_FRAME_RANGE_WORKERS, cpu_workers)
