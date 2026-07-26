"""Video Scan worker数を完了境界で安全に調整する。"""

from statistics import fmean

from ..models.video_scan_resource_sample import VideoScanResourceSample

_LOGICAL_CPUS_PER_CPU_WORKER = 8
_LOGICAL_CPUS_PER_NVDEC_WORKER = 4
_SPEED_SLOWDOWN_RATIO = 0.75
_ROLLING_SPEED_WINDOW = 4
_SAFE_WORKER_LIMIT = 32


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
            self._current_workers = (
                self._executor_capacity
                if _has_nvdec_headroom(initial_resource_sample)
                else self._conservative_workers
            )
        else:
            self._executor_capacity = self._conservative_workers
            self._current_workers = self._conservative_workers

        self._initial_workers = self._current_workers
        self._peak_workers = self._current_workers
        self._completed_scans = 0
        self._rolling_speeds: list[float] = []
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
        if self._mode == "fixed" or reused:
            return

        metrics: dict[str, object] = {}
        if resource_sample is not None:
            metrics.update(resource_sample.as_mapping())
        metrics["input_seconds_per_wall_second"] = input_seconds_per_wall_second
        previous_workers = self._current_workers
        reason = self._pressure_reason(
            resource_sample,
            input_seconds_per_wall_second,
        )
        if reason is not None and self._current_workers > 1:
            self._current_workers -= 1
        elif (
            reason is None
            and self._current_workers < self._executor_capacity
            and self._has_growth_headroom(resource_sample)
        ):
            self._current_workers += 1
            reason = (
                "gpu_headroom" if self._decode_backend == "nvdec" else "cpu_headroom"
            )

        self._record_speed(input_seconds_per_wall_second)
        if self._current_workers == previous_workers:
            return
        self._peak_workers = max(self._peak_workers, self._current_workers)
        self._changes.append(
            {
                "completed_scans": self._completed_scans,
                "from_workers": previous_workers,
                "to_workers": self._current_workers,
                "reason": reason,
                "metrics": metrics,
            }
        )

    def _pressure_reason(
        self,
        sample: VideoScanResourceSample | None,
        input_speed: float | None,
    ) -> str | None:
        if sample is None:
            return (
                "resource_sample_unavailable"
                if self._current_workers > self._conservative_workers
                else None
            )
        thresholds = (
            ("cpu_pressure", sample.cpu_percent, 90.0),
            (
                "cpu_core_pressure",
                sample.cpu_saturated_core_percent,
                30.0,
            ),
            ("memory_pressure", sample.memory_percent, 90.0),
            ("decoder_pressure", sample.decoder_percent, 92.0),
            ("gpu_pressure", sample.gpu_percent, 95.0),
            ("vram_pressure", sample.vram_percent, 90.0),
            ("disk_pressure", sample.disk_busy_percent, 92.0),
            ("disk_latency", sample.disk_read_latency_ms, 50.0),
        )
        for reason, value, threshold in thresholds:
            if value is not None and value >= threshold:
                return reason
        if (
            self._decode_backend == "nvdec"
            and self._current_workers > self._conservative_workers
            and not _nvdec_sample_complete(sample)
        ):
            return "resource_sample_incomplete"
        if (
            input_speed is not None
            and input_speed > 0
            and self._rolling_speeds
            and input_speed < fmean(self._rolling_speeds) * _SPEED_SLOWDOWN_RATIO
        ):
            return "stream_slowdown"
        return None

    def _has_growth_headroom(
        self,
        sample: VideoScanResourceSample | None,
    ) -> bool:
        if self._decode_backend == "nvdec":
            return _has_nvdec_headroom(sample)
        if sample is None:
            return False
        return (
            _below(sample.cpu_percent, 75.0)
            and _below_optional(sample.cpu_saturated_core_percent, 25.0)
            and _below(sample.memory_percent, 80.0)
            and _below_optional(sample.disk_busy_percent, 80.0)
            and _below_optional(sample.disk_read_latency_ms, 30.0)
        )

    def _record_speed(self, input_speed: float | None) -> None:
        if input_speed is None or input_speed <= 0:
            return
        self._rolling_speeds.append(input_speed)
        del self._rolling_speeds[:-_ROLLING_SPEED_WINDOW]


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
