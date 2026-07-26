"""Acceptance Run Attempt中のGPU resource sampler。"""

import json
import os
import shutil
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from threading import Event, Lock, Thread
from urllib.request import Request, urlopen

from ..models.processing_stage import ProcessingStage

GpuProbe = Callable[[], Mapping[str, int]]
StageProvider = Callable[[], ProcessingStage | None]

_VISION_STAGES = {
    ProcessingStage.BUILD_SCENE_CATALOG,
    ProcessingStage.ANNOTATE_CANDIDATE,
}
_MIB_BYTES = 1024**2


class GpuResourceMonitor:
    """system/process/model VRAMをStage別にsampleしてpeakを保持する。"""

    def __init__(
        self,
        *,
        ollama_host: str,
        stage_provider: StageProvider,
        probe: GpuProbe | None = None,
        interval_seconds: float = 0.5,
        join_timeout_seconds: float = 2.0,
    ) -> None:
        if interval_seconds <= 0 or join_timeout_seconds <= 0:
            raise ValueError("GPU sampler interval/停止timeoutは正の値が必要です")
        self._probe = probe or _default_probe(ollama_host)
        self._stage_provider = stage_provider
        self._interval_seconds = interval_seconds
        self._join_timeout_seconds = join_timeout_seconds
        self._stop = Event()
        self._lock = Lock()
        self._thread: Thread | None = None
        self._sample_count = 0
        self._sample_errors = 0
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
        if self._thread is not None:
            raise RuntimeError("GPU resource monitorは一度だけ開始できます")
        self._sample(is_baseline=True)
        self._thread = Thread(target=self._run, name="acceptance-gpu-monitor")
        self._thread.daemon = True
        self._thread.start()

    def stop(self) -> dict[str, object]:
        """samplerを停止しbudget用のsafe aggregateを返す。"""
        self._stop.set()
        sampler_stopped = False
        if self._thread is not None:
            self._thread.join(timeout=self._join_timeout_seconds)
            sampler_stopped = not self._thread.is_alive()
        if sampler_stopped:
            self._sample(is_baseline=False)
        with self._lock:
            return {
                "resource_sampling_complete": (
                    sampler_stopped
                    and self._sample_count > 0
                    and self._sample_errors == 0
                ),
                "gpu_sample_count": self._sample_count,
                "gpu_sample_error_count": self._sample_errors,
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
        self._sample(is_baseline=False)

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            self._sample(is_baseline=False)

    def _sample(self, *, is_baseline: bool) -> None:
        try:
            sample = self._probe()
            system = _non_negative_integer(sample, "system_used_mib")
            process = _non_negative_integer(sample, "process_used_mib")
            model_size = _non_negative_integer(sample, "ollama_size_bytes")
            model_vram = _non_negative_integer(sample, "ollama_size_vram_bytes")
        except Exception:
            with self._lock:
                self._sample_errors += 1
            return
        stage = self._stage_provider()
        with self._lock:
            self._sample_count += 1
            if is_baseline:
                self._process_baseline_mib = process
                self._system_baseline_mib = system
            self._system_peak_mib = max(self._system_peak_mib, system)
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
            if stage in _VISION_STAGES:
                self._ollama_peak_mib = max(self._ollama_peak_mib, system)
            elif stage is ProcessingStage.COLLECT_CONTEXT:
                non_ollama_system = max(
                    system - model_vram // _MIB_BYTES,
                    0,
                )
                self._stt_peak_mib = max(
                    self._stt_peak_mib,
                    non_ollama_system,
                )


def _default_probe(ollama_host: str) -> GpuProbe:
    command = find_nvidia_smi()

    def probe() -> Mapping[str, int]:
        system = _query_integer(
            [command, "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        )
        process = _query_current_process_memory(command)
        model_size, model_vram = _query_ollama_sizes(ollama_host)
        return {
            "system_used_mib": system,
            "process_used_mib": process,
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
    process = subprocess.run(command, check=True, capture_output=True, text=True)
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
