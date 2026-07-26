"""Video Scan並列制御用のprivacy-safe resource sample。"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class VideoScanResourceSample:
    """worker判断に必要な割合とthroughputだけを保持する。"""

    cpu_percent: float | None
    memory_percent: float | None
    decoder_percent: float | None
    gpu_percent: float | None
    vram_percent: float | None
    disk_busy_percent: float | None
    disk_read_mib_per_second: float | None
    disk_read_latency_ms: float | None = None
    cpu_saturated_core_percent: float | None = None

    def __post_init__(self) -> None:
        """割合とthroughputが有限かつ非負であることを検証する。"""
        for name in (
            "cpu_percent",
            "memory_percent",
            "decoder_percent",
            "gpu_percent",
            "vram_percent",
            "disk_busy_percent",
            "cpu_saturated_core_percent",
        ):
            value = getattr(self, name)
            if value is not None and (
                not math.isfinite(value) or not 0 <= value <= 100
            ):
                raise ValueError(f"{name}は0以上100以下の有限値が必要です")
        throughput = self.disk_read_mib_per_second
        if throughput is not None and (not math.isfinite(throughput) or throughput < 0):
            raise ValueError("disk_read_mib_per_secondは非負の有限値が必要です")
        latency = self.disk_read_latency_ms
        if latency is not None and (not math.isfinite(latency) or latency < 0):
            raise ValueError("disk_read_latency_msは非負の有限値が必要です")

    def as_mapping(self) -> dict[str, float | None]:
        """pathやdevice名を含まないreport用mappingを返す。"""
        return {
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "decoder_percent": self.decoder_percent,
            "gpu_percent": self.gpu_percent,
            "vram_percent": self.vram_percent,
            "disk_busy_percent": self.disk_busy_percent,
            "disk_read_mib_per_second": self.disk_read_mib_per_second,
            "disk_read_latency_ms": self.disk_read_latency_ms,
            "cpu_saturated_core_percent": self.cpu_saturated_core_percent,
        }
