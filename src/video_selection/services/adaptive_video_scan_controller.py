"""Video Scan worker数を完了境界で安全に調整する。"""

import time
from collections.abc import Callable
from statistics import fmean

from ..models.video_scan_resource_sample import VideoScanResourceSample

_LOGICAL_CPUS_PER_CPU_WORKER = 8
_LOGICAL_CPUS_PER_NVDEC_WORKER = 4
_SPEED_SLOWDOWN_RATIO = 0.75
_DISK_THROUGHPUT_SLOWDOWN_RATIO = 0.65
_RESOURCE_WINDOW = 3
_TREND_HALF_WINDOW = 2
_TREND_WINDOW = _TREND_HALF_WINDOW * 2
_SAFE_WORKER_LIMIT = 32
Clock = Callable[[], float]


class AdaptiveVideoScanController:
    """設定とresource sampleからbounded worker数を決定する深いmodule。"""

    def __init__(
        self,
        *,
        video_count: int,
        configured_workers: str | int,
        auto_max_workers: int,
        decode_backend: str,
        logical_cpu_count: int,
        initial_resource_sample: VideoScanResourceSample | None,
        clock: Clock = time.monotonic,
    ) -> None:
        if video_count < 1 or auto_max_workers < 1 or logical_cpu_count < 1:
            raise ValueError("Video Scan worker境界には正の件数が必要です")
        if auto_max_workers > _SAFE_WORKER_LIMIT or (
            type(configured_workers) is int and configured_workers > _SAFE_WORKER_LIMIT
        ):
            raise ValueError("Video Scan worker境界は32以下である必要があります")
        if decode_backend not in {"cpu", "nvdec"}:
            raise ValueError("Video Scan decode backendが不正です")
        if configured_workers != "auto" and (
            type(configured_workers) is not int or configured_workers < 1
        ):
            raise ValueError("Video Scan workersはautoまたは正の整数が必要です")

        self._mode = "auto" if configured_workers == "auto" else "fixed"
        self._video_count = video_count
        self._configured_workers = configured_workers
        self._decode_backend = decode_backend
        self._auto_max_workers = auto_max_workers
        self._conservative_workers = min(
            video_count,
            auto_max_workers,
            max(1, logical_cpu_count // _LOGICAL_CPUS_PER_CPU_WORKER),
        )
        nvdec_capacity = min(
            video_count,
            auto_max_workers,
            max(1, logical_cpu_count // _LOGICAL_CPUS_PER_NVDEC_WORKER),
        )
        if self._mode == "fixed":
            fixed_workers = min(video_count, int(configured_workers))
            self._executor_capacity = fixed_workers
            self._current_workers = fixed_workers
        elif decode_backend == "nvdec":
            self._executor_capacity = max(
                self._conservative_workers,
                nvdec_capacity,
            )
            self._current_workers = _initial_auto_workers(
                self._conservative_workers,
                decode_backend,
                initial_resource_sample,
            )
        else:
            self._executor_capacity = self._conservative_workers
            self._current_workers = _initial_auto_workers(
                self._conservative_workers,
                decode_backend,
                initial_resource_sample,
            )

        self._initial_workers = self._current_workers
        self._peak_workers = self._current_workers
        self._completed_scans = 0
        self._clock = clock
        self._started_at = clock()
        self._scan_wall_seconds = 0.0
        self._resource_samples: list[VideoScanResourceSample] = []
        self._rolling_speeds: list[float] = []
        self._disk_throughputs: list[float] = []
        self._changes: list[dict[str, object]] = []
        self._initial_metrics = (
            None
            if initial_resource_sample is None
            else initial_resource_sample.as_mapping()
        )

    @property
    def current_workers(self) -> int:
        """次のtask admissionに利用する現在worker数を返す。"""
        return self._current_workers

    @property
    def executor_capacity(self) -> int:
        """このrunで利用し得るThreadPool容量を返す。"""
        return self._executor_capacity

    @property
    def resource_sampling_enabled(self) -> bool:
        """worker数を自動調整するrunかを返す。"""
        return self._mode == "auto"

    @property
    def diagnostics(self) -> dict[str, object]:
        """cache identityへ含めないprivacy-safeなrun診断を返す。"""
        return {
            "mode": self._mode,
            "configured_workers": self._configured_workers,
            "decode_backend": self._decode_backend,
            "auto_max_workers": self._auto_max_workers,
            "initial_workers": self._initial_workers,
            "final_workers": self._current_workers,
            "peak_workers": self._peak_workers,
            "completed_scans": self._completed_scans,
            "scan_wall_seconds": self._scan_wall_seconds,
            "initial_metrics": self._initial_metrics,
            "changes": [dict(change) for change in self._changes],
        }

    def observe_scan_completion(
        self,
        *,
        reused: bool,
        input_seconds_per_wall_second: float | None,
        resource_sample: VideoScanResourceSample | None,
    ) -> None:
        """一つの完了境界でworker数を最大1だけ変更する。"""
        self._completed_scans += 1
        observed_at = self._clock() if not reused else None
        if observed_at is not None:
            self._scan_wall_seconds = round(
                max(0.0, observed_at - self._started_at),
                3,
            )
        if self._mode == "fixed" or reused:
            return

        self._record_resource_sample(resource_sample)
        self._record_speed(input_seconds_per_wall_second)
        rolling_sample = self._rolling_resource_sample()
        metrics = self._decision_metrics(
            resource_sample,
            rolling_sample,
            input_seconds_per_wall_second,
        )
        previous_workers = self._current_workers
        reason = self._pressure_reason(
            resource_sample,
            rolling_sample,
        )
        if reason is not None and self._current_workers > 1:
            self._current_workers -= 1
        elif (
            reason is None
            and self._current_workers < self._executor_capacity
            and self._remaining_scans_can_fill_growth()
            and self._has_growth_headroom(rolling_sample)
        ):
            self._current_workers += 1
            reason = (
                "gpu_headroom" if self._decode_backend == "nvdec" else "cpu_headroom"
            )

        if self._current_workers == previous_workers:
            return
        self._peak_workers = max(self._peak_workers, self._current_workers)
        self._changes.append(
            {
                "completed_scans": self._completed_scans,
                "from_workers": previous_workers,
                "to_workers": self._current_workers,
                "reason": reason,
                "elapsed_seconds": self._scan_wall_seconds,
                "metrics": metrics,
            }
        )

    def _remaining_scans_can_fill_growth(self) -> bool:
        """増加後のworker数を未完了scanで実際に満たせるか返す。"""
        remaining_scans = self._video_count - self._completed_scans
        return remaining_scans >= self._current_workers + 1

    def finish_incomplete_attempt(self) -> None:
        """未完了scanを停止し終えた時点までのattempt wall時間を確定する。"""
        if self._completed_scans >= self._video_count:
            return
        self._scan_wall_seconds = round(
            max(0.001, self._clock() - self._started_at),
            3,
        )

    def _pressure_reason(
        self,
        sample: VideoScanResourceSample | None,
        rolling_sample: VideoScanResourceSample | None,
    ) -> str | None:
        if sample is None:
            return (
                "resource_sample_unavailable"
                if self._current_workers > self._conservative_workers
                else None
            )
        if (
            self._decode_backend == "nvdec"
            and self._current_workers > self._conservative_workers
            and not _nvdec_sample_complete(sample)
        ):
            return "resource_sample_incomplete"
        if rolling_sample is None:
            return None
        resource_reason = _resource_pressure_reason(
            rolling_sample,
            self._decode_backend,
        )
        if resource_reason is not None:
            return resource_reason
        disk_ratio = _trend_ratio(self._disk_throughputs)
        if (
            disk_ratio is not None
            and disk_ratio < _DISK_THROUGHPUT_SLOWDOWN_RATIO
            and (
                _at_least(rolling_sample.disk_busy_percent, 60.0)
                or _at_least(rolling_sample.disk_read_latency_ms, 20.0)
            )
        ):
            return "disk_throughput_slowdown"
        speed_ratio = _trend_ratio(self._rolling_speeds)
        if speed_ratio is not None and speed_ratio < _SPEED_SLOWDOWN_RATIO:
            return "stream_slowdown"
        return None

    def _has_growth_headroom(
        self,
        sample: VideoScanResourceSample | None,
    ) -> bool:
        if sample is None:
            return False
        disk_ratio = _trend_ratio(self._disk_throughputs)
        speed_ratio = _trend_ratio(self._rolling_speeds)
        if speed_ratio is None or speed_ratio < _SPEED_SLOWDOWN_RATIO:
            return False
        if disk_ratio is not None and disk_ratio < _DISK_THROUGHPUT_SLOWDOWN_RATIO:
            return False
        if not all(_has_disk_observation(item) for item in self._resource_samples):
            return False
        if self._decode_backend == "nvdec":
            return _has_nvdec_headroom(sample) and all(
                _has_nvdec_headroom(item) for item in self._resource_samples
            )
        return _has_cpu_headroom(sample) and all(
            _has_cpu_headroom(item) for item in self._resource_samples
        )

    def _record_speed(self, input_speed: float | None) -> None:
        if input_speed is None or input_speed <= 0:
            self._rolling_speeds.clear()
            return
        self._rolling_speeds.append(input_speed)
        del self._rolling_speeds[:-_TREND_WINDOW]

    def _record_resource_sample(
        self,
        sample: VideoScanResourceSample | None,
    ) -> None:
        if sample is None:
            self._resource_samples.clear()
            self._disk_throughputs.clear()
            return
        self._resource_samples.append(sample)
        del self._resource_samples[:-_RESOURCE_WINDOW]
        throughput = sample.disk_read_mib_per_second
        if throughput is None:
            return
        self._disk_throughputs.append(throughput)
        del self._disk_throughputs[:-_TREND_WINDOW]

    def _rolling_resource_sample(self) -> VideoScanResourceSample | None:
        if len(self._resource_samples) < _RESOURCE_WINDOW:
            return None
        return _average_resource_samples(self._resource_samples)

    def _decision_metrics(
        self,
        latest_sample: VideoScanResourceSample | None,
        rolling_sample: VideoScanResourceSample | None,
        input_speed: float | None,
    ) -> dict[str, object]:
        return {
            "latest": (None if latest_sample is None else latest_sample.as_mapping()),
            "rolling": (
                None if rolling_sample is None else rolling_sample.as_mapping()
            ),
            "input_seconds_per_wall_second": input_speed,
            "rolling_input_seconds_per_wall_second": (
                fmean(self._rolling_speeds) if self._rolling_speeds else None
            ),
            "disk_throughput_ratio": _trend_ratio(self._disk_throughputs),
            "stream_speed_ratio": _trend_ratio(self._rolling_speeds),
        }


def _average_resource_samples(
    samples: list[VideoScanResourceSample],
) -> VideoScanResourceSample:
    return VideoScanResourceSample(
        cpu_percent=_average_metric(samples, "cpu_percent"),
        memory_percent=_average_metric(samples, "memory_percent"),
        decoder_percent=_average_metric(samples, "decoder_percent"),
        gpu_percent=_average_metric(samples, "gpu_percent"),
        vram_percent=_average_metric(samples, "vram_percent"),
        disk_busy_percent=_average_metric(samples, "disk_busy_percent"),
        disk_read_mib_per_second=_average_metric(
            samples,
            "disk_read_mib_per_second",
        ),
        disk_read_latency_ms=_average_metric(samples, "disk_read_latency_ms"),
        cpu_saturated_core_percent=_average_metric(
            samples,
            "cpu_saturated_core_percent",
        ),
    )


def _average_metric(
    samples: list[VideoScanResourceSample],
    name: str,
) -> float | None:
    values = tuple(
        value for sample in samples if (value := getattr(sample, name)) is not None
    )
    return fmean(values) if values else None


def _trend_ratio(values: list[float]) -> float | None:
    if len(values) < _TREND_WINDOW:
        return None
    baseline = fmean(values[-_TREND_WINDOW:-_TREND_HALF_WINDOW])
    if baseline <= 0:
        return None
    recent = fmean(values[-_TREND_HALF_WINDOW:])
    return recent / baseline


def _has_nvdec_headroom(sample: VideoScanResourceSample | None) -> bool:
    if sample is None or not _nvdec_sample_complete(sample):
        return False
    return (
        _below(sample.cpu_percent, 80.0)
        and _below_optional(sample.cpu_saturated_core_percent, 25.0)
        and _below(sample.memory_percent, 85.0)
        and _below(sample.decoder_percent, 78.0)
        and _below(sample.gpu_percent, 85.0)
        and _below(sample.vram_percent, 85.0)
        and _below_optional(sample.disk_busy_percent, 85.0)
        and _below_optional(sample.disk_read_latency_ms, 30.0)
    )


def _has_cpu_headroom(sample: VideoScanResourceSample) -> bool:
    return (
        _below(sample.cpu_percent, 75.0)
        and _below_optional(sample.cpu_saturated_core_percent, 25.0)
        and _below(sample.memory_percent, 80.0)
        and _below_optional(sample.disk_busy_percent, 80.0)
        and _below_optional(sample.disk_read_latency_ms, 30.0)
    )


def _has_disk_observation(sample: VideoScanResourceSample) -> bool:
    return sample.disk_busy_percent is not None and (
        sample.disk_read_mib_per_second is not None
        or sample.disk_read_latency_ms is not None
    )


def _initial_auto_workers(
    conservative_workers: int,
    decode_backend: str,
    sample: VideoScanResourceSample | None,
) -> int:
    if sample is None:
        return conservative_workers
    if _resource_pressure_reason(sample, decode_backend) is not None:
        return 1
    return conservative_workers


def _resource_pressure_reason(
    sample: VideoScanResourceSample,
    decode_backend: str,
) -> str | None:
    thresholds = [
        ("cpu_pressure", sample.cpu_percent, 90.0),
        (
            "cpu_core_pressure",
            sample.cpu_saturated_core_percent,
            30.0,
        ),
        ("memory_pressure", sample.memory_percent, 90.0),
        ("disk_pressure", sample.disk_busy_percent, 92.0),
        ("disk_latency", sample.disk_read_latency_ms, 50.0),
    ]
    if decode_backend == "nvdec":
        thresholds.extend(
            (
                ("decoder_pressure", sample.decoder_percent, 92.0),
                ("gpu_pressure", sample.gpu_percent, 95.0),
                ("vram_pressure", sample.vram_percent, 90.0),
            )
        )
    for reason, value, threshold in thresholds:
        if _at_least(value, threshold):
            return reason
    return None


def _nvdec_sample_complete(sample: VideoScanResourceSample) -> bool:
    return all(
        value is not None
        for value in (
            sample.cpu_percent,
            sample.memory_percent,
            sample.decoder_percent,
            sample.gpu_percent,
            sample.vram_percent,
        )
    )


def _below(value: float | None, threshold: float) -> bool:
    return value is not None and value < threshold


def _below_optional(value: float | None, threshold: float) -> bool:
    return value is None or value < threshold


def _at_least(value: float | None, threshold: float) -> bool:
    return value is not None and value >= threshold
