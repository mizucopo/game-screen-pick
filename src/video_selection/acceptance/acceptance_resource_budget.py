"""Target acceptanceで共有するresource予算判定。"""

from collections.abc import Mapping

PERSISTENT_CACHE_BYTES = 64 * 1024**3
PEAK_ADDITIONAL_BYTES = 96 * 1024**3
OLLAMA_GPU_MIB = 18 * 1024
STT_GPU_MIB = 8 * 1024


def phase_resource_budget_passed(record: Mapping[str, object]) -> bool:
    """一つのphaseが計測完全性と共有resource予算を満たすか返す。"""
    return (
        _boolean(record, "resource_sampling_complete")
        and _integer(record, "persistent_cache_bytes") <= PERSISTENT_CACHE_BYTES
        and _integer(record, "peak_additional_bytes") <= PEAK_ADDITIONAL_BYTES
        and _integer(record, "ollama_global_gpu_peak_mib") <= OLLAMA_GPU_MIB
        and _integer(record, "stt_non_ollama_gpu_peak_mib") <= STT_GPU_MIB
    )


def _integer(value: Mapping[str, object], key: str) -> int:
    result = value.get(key)
    if not isinstance(result, int) or isinstance(result, bool):
        raise ValueError(f"Acceptance phase metric {key}がintegerではありません")
    return result


def _boolean(value: Mapping[str, object], key: str) -> bool:
    result = value.get(key)
    if not isinstance(result, bool):
        raise ValueError(f"Acceptance phase metric {key}がbooleanではありません")
    return result
