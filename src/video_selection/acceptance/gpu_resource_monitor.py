"""Acceptance Run Attempt中のGPU resource sampler。"""

import json
import os
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from threading import Event, Lock, Thread
from urllib.request import Request, urlopen

from ..models.processing_stage import ProcessingStage

SystemGpuProbe = Callable[[], Mapping[str, int]]
OllamaGpuProbe = Callable[[], Mapping[str, int]]
StageProvider = Callable[[], ProcessingStage | None]

_VISION_STAGES = {
    ProcessingStage.BUILD_SCENE_CATALOG,
    ProcessingStage.ANNOTATE_CANDIDATE,
}
_MIB_BYTES = 1024**2
_NVIDIA_SMI_TIMEOUT_SECONDS = 2


class GpuResourceMonitor:
    """system/process/model VRAMをStage別にsampleしてpeakを保持する。"""

    def __init__(
        self,
        *,
        ollama_host: str,
        stage_provider: StageProvider,
        system_probe: SystemGpuProbe | None = None,
        ollama_probe: OllamaGpuProbe | None = None,
        interval_seconds: float = 0.5,
        join_timeout_seconds: float = 2.0,
    ) -> None:
        if interval_seconds <= 0 or join_timeout_seconds <= 0:
            raise ValueError("GPU sampler interval/停止timeoutは正の値が必要です")
        self._system_probe = system_probe or _default_system_probe()
        self._ollama_probe = ollama_probe or _default_ollama_probe(ollama_host)
        self._stage_provider = stage_provider
        self._interval_seconds = interval_seconds
        self._join_timeout_seconds = join_timeout_seconds
        self._stop = Event()
        self._lock = Lock()
        self._ollama_probe_lock = Lock()
        self._system_thread: Thread | None = None
        self._ollama_thread: Thread | None = None
        self._system_sample_count = 0
        self._system_sample_errors = 0
        self._ollama_sample_count = 0
        self._ollama_sample_errors = 0
        self._ollama_stop_timed_out = False
        self._current_ollama_vram_mib: int | None = None
        self._process_baseline_mib = 0
        self._system_baseline_mib = 0
        self._system_peak_mib = 0
        self._ollama_peak_mib = 0
        self._stt_peak_mib = 0
        self._ollama_size_bytes = 0
        self._ollama_size_vram_bytes = 0
        self._ollama_model_observed = False
        self._ollama_model_fully_resident = True

    def start(self) -> None:
        """baselineをsampleしてbackground samplerを開始する。"""
        if self._system_thread is not None or self._ollama_thread is not None:
            raise RuntimeError("GPU resource monitorは一度だけ開始できます")
        self._record_system_sample(is_baseline=True)
        self._system_thread = Thread(
            target=self._run_system_sampler,
            name="acceptance-system-gpu-monitor",
            daemon=True,
        )
        self._ollama_thread = Thread(
            target=self._run_ollama_sampler,
            name="acceptance-ollama-gpu-monitor",
            daemon=True,
        )
        self._system_thread.start()
        self._ollama_thread.start()

    def stop(self) -> dict[str, object]:
        """samplerを停止しbudget用のsafe aggregateを返す。"""
        self._stop.set()
        deadline = time.monotonic() + self._join_timeout_seconds
        system_sampler_stopped = self._join_before_deadline(
            self._system_thread,
            deadline,
        )
        ollama_sampler_stopped = self._join_before_deadline(
            self._ollama_thread,
            deadline,
        )
        if system_sampler_stopped:
            self._record_system_sample(is_baseline=False)
        with self._lock:
            if (
                self._ollama_thread is not None
                and not ollama_sampler_stopped
                and not self._ollama_stop_timed_out
            ):
                self._ollama_stop_timed_out = True
                self._ollama_sample_errors += 1
                self._current_ollama_vram_mib = None
            return {
                "resource_sampling_complete": (
                    system_sampler_stopped
                    and self._system_sample_count > 0
                    and self._system_sample_errors == 0
                ),
                "gpu_sample_count": self._system_sample_count,
                "gpu_sample_error_count": self._system_sample_errors,
                "ollama_sample_count": self._ollama_sample_count,
                "ollama_sample_error_count": self._ollama_sample_errors,
                "process_gpu_baseline_mib": self._process_baseline_mib,
                "system_gpu_baseline_mib": self._system_baseline_mib,
                "system_global_gpu_peak_mib": self._system_peak_mib,
                "ollama_global_gpu_peak_mib": self._ollama_peak_mib,
                "stt_non_ollama_gpu_peak_mib": self._stt_peak_mib,
                "ollama_model_size_bytes": self._ollama_size_bytes,
                "ollama_model_size_vram_bytes": self._ollama_size_vram_bytes,
                "ollama_model_observed": self._ollama_model_observed,
                "ollama_model_fully_resident": (
                    self._ollama_model_observed and self._ollama_model_fully_resident
                ),
            }

    def sample_now(self) -> None:
        """target preflightやdeterministic testから即時sampleを要求する。"""
        self._record_ollama_sample()
        self._record_system_sample(is_baseline=False)

    def _run_system_sampler(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            self._record_system_sample(is_baseline=False)

    def _run_ollama_sampler(self) -> None:
        self._record_ollama_sample()
        while not self._stop.wait(self._interval_seconds):
            self._record_ollama_sample()

    def _record_system_sample(self, *, is_baseline: bool) -> None:
        system_sample = self._sample_system()
        stage = self._stage_provider()
        with self._lock:
            if system_sample is None:
                self._system_sample_errors += 1
            else:
                system, process = system_sample
                self._system_sample_count += 1
                if is_baseline:
                    self._process_baseline_mib = process
                    self._system_baseline_mib = system
                self._system_peak_mib = max(self._system_peak_mib, system)
                if not is_baseline and stage in _VISION_STAGES:
                    self._ollama_peak_mib = max(self._ollama_peak_mib, system)
                elif not is_baseline and stage is ProcessingStage.COLLECT_CONTEXT:
                    non_ollama_system = max(
                        system - (self._current_ollama_vram_mib or 0),
                        0,
                    )
                    self._stt_peak_mib = max(
                        self._stt_peak_mib,
                        non_ollama_system,
                    )

    def _record_ollama_sample(self) -> None:
        with self._ollama_probe_lock:
            with self._lock:
                if self._ollama_stop_timed_out:
                    return
                self._current_ollama_vram_mib = None
            ollama_sample = self._sample_ollama()
            with self._lock:
                if self._ollama_stop_timed_out:
                    return
                if ollama_sample is None:
                    self._ollama_sample_errors += 1
                    return
                model_size, model_vram = ollama_sample
                self._ollama_sample_count += 1
                self._current_ollama_vram_mib = model_vram // _MIB_BYTES
                self._ollama_size_bytes = max(
                    self._ollama_size_bytes,
                    model_size,
                )
                self._ollama_size_vram_bytes = max(
                    self._ollama_size_vram_bytes,
                    model_vram,
                )
                if model_size > 0:
                    self._ollama_model_observed = True
                    self._ollama_model_fully_resident = (
                        self._ollama_model_fully_resident and model_vram == model_size
                    )

    @staticmethod
    def _join_before_deadline(thread: Thread | None, deadline: float) -> bool:
        if thread is None:
            return False
        thread.join(timeout=max(0.0, deadline - time.monotonic()))
        return not thread.is_alive()

    def _sample_system(self) -> tuple[int, int] | None:
        for attempt in range(2):
            try:
                sample = self._system_probe()
                system = _non_negative_integer(sample, "system_used_mib")
                process = _non_negative_integer(sample, "process_used_mib")
                return system, process
            except Exception:
                if attempt == 0:
                    continue
        return None

    def _sample_ollama(self) -> tuple[int, int] | None:
        for attempt in range(2):
            try:
                sample = self._ollama_probe()
                model_size = _non_negative_integer(sample, "ollama_size_bytes")
                model_vram = _non_negative_integer(
                    sample,
                    "ollama_size_vram_bytes",
                )
                return model_size, model_vram
            except Exception:
                if attempt == 0:
                    continue
        return None


def _default_system_probe() -> SystemGpuProbe:
    command = find_nvidia_smi()
    process_baseline: int | None = None

    def probe() -> Mapping[str, int]:
        nonlocal process_baseline
        system = _query_integer(
            [command, "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        )
        if process_baseline is None:
            process_baseline = _query_current_process_memory(command)
        return {
            "system_used_mib": system,
            "process_used_mib": process_baseline,
        }

    return probe


def _default_ollama_probe(ollama_host: str) -> OllamaGpuProbe:
    def probe() -> Mapping[str, int]:
        model_size, model_vram = _query_ollama_sizes(ollama_host)
        return {
            "ollama_size_bytes": model_size,
            "ollama_size_vram_bytes": model_vram,
        }

    return probe


def find_nvidia_smi() -> str:
    """PATHまたはWSL既定locationからnvidia-smiを返す。"""
    executable = shutil.which("nvidia-smi")
    if executable is not None:
        return executable
    wsl_executable = Path("/usr/lib/wsl/lib/nvidia-smi")
    if wsl_executable.is_file():
        return str(wsl_executable)
    raise ValueError("nvidia-smiが見つかりません")


def _query_integer(command: list[str]) -> int:
    process = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=_NVIDIA_SMI_TIMEOUT_SECONDS,
    )
    values = [
        int(line.strip())
        for line in process.stdout.splitlines()
        if line.strip().isdigit()
    ]
    if not values:
        raise ValueError("nvidia-smiがGPU memoryを返しませんでした")
    return sum(values)


def _query_current_process_memory(command: str) -> int:
    process = subprocess.run(
        [
            command,
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=_NVIDIA_SMI_TIMEOUT_SECONDS,
    )
    total = 0
    for line in process.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if (
            len(parts) == 2
            and parts[0].isdigit()
            and parts[1].isdigit()
            and int(parts[0]) == os.getpid()
        ):
            total += int(parts[1])
    return total


def _query_ollama_sizes(host: str) -> tuple[int, int]:
    request = Request(f"{host.rstrip('/')}/api/ps", method="GET")
    with urlopen(request, timeout=5) as response:  # noqa: S310 - configured target
        value: object = json.loads(response.read())
    if not isinstance(value, dict) or not isinstance(value.get("models"), list):
        raise ValueError("Ollama /api/ps responseが不正です")
    total_size = 0
    total_vram = 0
    for item in value["models"]:
        if isinstance(item, dict):
            size = item.get("size", 0)
            size_vram = item.get("size_vram", 0)
            if (
                not isinstance(size, int)
                or isinstance(size, bool)
                or size < 0
                or not isinstance(size_vram, int)
                or isinstance(size_vram, bool)
                or size_vram < 0
            ):
                raise ValueError("Ollama /api/ps model sizeが不正です")
            total_size += size
            total_vram += size_vram
    return total_size, total_vram


def _non_negative_integer(value: Mapping[str, int], key: str) -> int:
    result = value.get(key)
    if not isinstance(result, int) or isinstance(result, bool) or result < 0:
        raise ValueError(f"GPU sample {key}が不正です")
    return result
