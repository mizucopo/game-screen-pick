"""privacy-safe target acceptance recordとnormalized baseline。"""

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from uuid import uuid4

from ..models.report_value import string_looks_private
from .acceptance_resource_budget import (
    OLLAMA_GPU_MIB,
    PEAK_ADDITIONAL_BYTES,
    PERSISTENT_CACHE_BYTES,
    STT_GPU_MIB,
)
from .atomic_json import write_atomic_json

_RECORD_SCHEMA = "game-screen-pick/target-acceptance@1.3.0"
_BASELINE_SCHEMA = "game-screen-pick/target-acceptance-baseline@1.3.0"
_DENIED_KEY_PARTS = (
    "absolute_path",
    "credential",
    "environment_variable",
    "input_root",
    "prompt",
    "raw_",
    "relative_video",
    "response_body",
    "secret",
    "source_name",
    "stack_trace",
)
_WINDOWS_PATH = re.compile(r"[A-Za-z]:[\\/]")

_BUDGETS: dict[str, dict[str, int]] = {
    "release": {
        "cold_seconds": 20 * 60,
        "warm_seconds": 3 * 60,
    },
    "full": {
        "cold_seconds": 24 * 60 * 60,
        "warm_seconds": 30 * 60,
    },
}


def build_acceptance_record(
    *,
    suite: str,
    commit: str,
    dirty: bool,
    target: Mapping[str, object],
    configuration: Mapping[str, object],
    models: Mapping[str, object],
    storage_preflight: Mapping[str, object],
    video_set: Mapping[str, object],
    cold: Mapping[str, object],
    warm: Mapping[str, object],
    human_quality: Mapping[str, object],
    video_scan_parallelism_comparison: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """phase metricsとquality aggregateからversioned acceptance recordを作る。"""
    if suite not in _BUDGETS:
        raise ValueError("Acceptance suiteが不正です")
    budgets = {
        **_BUDGETS[suite],
        "warm_unexpected_recompute": 0,
        "persistent_cache_bytes": PERSISTENT_CACHE_BYTES,
        "peak_additional_bytes": PEAK_ADDITIONAL_BYTES,
        "ollama_global_gpu_peak_mib": OLLAMA_GPU_MIB,
        "stt_non_ollama_gpu_peak_mib": STT_GPU_MIB,
    }
    consistency = cold.get("normalized_result_digest") == warm.get(
        "normalized_result_digest"
    )
    cold_speech_runtime = _optional_string(cold, "speech_runtime_identity")
    warm_speech_runtime = _optional_string(warm, "speech_runtime_identity")
    speech_runtime_consistency = cold_speech_runtime == warm_speech_runtime
    automatic_gates = {
        "cold_duration": _number(cold, "duration_seconds") <= budgets["cold_seconds"],
        "warm_duration": _number(warm, "duration_seconds") <= budgets["warm_seconds"],
        "warm_unexpected_recompute": _integer(warm, "unexpected_recompute_count") == 0,
        "warm_result_consistency": consistency,
        "speech_runtime_identity_consistency": speech_runtime_consistency,
        "resource_sampling": _boolean(cold, "resource_sampling_complete")
        and _boolean(warm, "resource_sampling_complete"),
        "persistent_cache": max(
            _integer(cold, "persistent_cache_bytes"),
            _integer(warm, "persistent_cache_bytes"),
        )
        <= PERSISTENT_CACHE_BYTES,
        "peak_additional_storage": max(
            _integer(cold, "peak_additional_bytes"),
            _integer(warm, "peak_additional_bytes"),
        )
        <= PEAK_ADDITIONAL_BYTES,
        "ollama_global_gpu_peak": max(
            _integer(cold, "ollama_global_gpu_peak_mib"),
            _integer(warm, "ollama_global_gpu_peak_mib"),
        )
        <= OLLAMA_GPU_MIB,
        "stt_non_ollama_gpu_peak": max(
            _integer(cold, "stt_non_ollama_gpu_peak_mib"),
            _integer(warm, "stt_non_ollama_gpu_peak_mib"),
        )
        <= STT_GPU_MIB,
        "ollama_model_fully_resident": (
            _boolean(cold, "ollama_model_observed")
            and _boolean(cold, "ollama_model_fully_resident")
            and (
                not _boolean(warm, "ollama_model_observed")
                or _boolean(warm, "ollama_model_fully_resident")
            )
        ),
    }
    if suite == "full":
        if video_scan_parallelism_comparison is None:
            raise ValueError("Full acceptanceにVideo Scan parallelism比較がありません")
        comparison_gates = video_scan_parallelism_comparison.get("gates")
        if not isinstance(comparison_gates, Mapping):
            raise ValueError("Video Scan parallelism comparison gateが不正です")
        automatic_gates.update(
            {
                f"video_scan_{name}": _comparison_gate(comparison_gates, name)
                for name in (
                    "execution_context_equal",
                    "fixed_three_workers",
                    "auto_exceeded_three_workers",
                    "stage_artifacts_equal",
                    "resource_budget",
                    "wall_time_improved",
                )
            }
        )
    human_status = human_quality.get("status")
    automatic_passed = all(automatic_gates.values())
    status = (
        "failed"
        if not automatic_passed or human_status == "failed"
        else "passed"
        if human_status == "passed"
        else "pending_human_review"
    )
    record: dict[str, object] = {
        "schema": _RECORD_SCHEMA,
        "status": status,
        "suite": suite,
        "source_revision": {"commit": commit, "dirty": dirty},
        "target": dict(target),
        "configuration": dict(configuration),
        "models": dict(models),
        "runtime": {"speech_to_text": cold_speech_runtime},
        "storage_preflight": dict(storage_preflight),
        "video_set": dict(video_set),
        "phases": {"cold": dict(cold), "warm": dict(warm)},
        "consistency": {
            "normalized_result_equal": consistency,
            "speech_runtime_identity_equal": speech_runtime_consistency,
        },
        "budgets": budgets,
        "automatic_gates": automatic_gates,
        "human_quality": dict(human_quality),
        "privacy": {
            "actual_paths": "omitted",
            "video_names": "omitted",
            "media_text": "omitted",
            "model_io": "omitted",
        },
    }
    if video_scan_parallelism_comparison is not None:
        record["video_scan_parallelism_comparison"] = dict(
            video_scan_parallelism_comparison
        )
    return record


def validate_acceptance_record_privacy(
    record: Mapping[str, object],
    *,
    forbidden_values: Sequence[str],
) -> None:
    """actual path/nameとsecret-bearing fieldがrecordへ混入していないことを検証する。"""
    _validate_private_value(record, forbidden_values, "record")


def write_normalized_baseline(
    record: Mapping[str, object],
    directory: Path,
) -> tuple[Path, Path]:
    """承認済みrecordからcommit可能なnormalized JSON/Markdownを生成する。"""
    if record.get("status") != "passed":
        raise ValueError("合格済みacceptance recordだけをbaseline化できます")
    normalized_value = _normalize_baseline_value(
        {
            "schema": _BASELINE_SCHEMA,
            "suite": record.get("suite"),
            "target": record.get("target"),
            "configuration": record.get("configuration"),
            "models": record.get("models"),
            "runtime": record.get("runtime"),
            "storage_preflight": record.get("storage_preflight"),
            "video_set": record.get("video_set"),
            "phases": record.get("phases"),
            "budgets": record.get("budgets"),
            "automatic_gates": record.get("automatic_gates"),
            "video_scan_parallelism_comparison": record.get(
                "video_scan_parallelism_comparison"
            ),
            "human_quality": record.get("human_quality"),
        },
        within_execution_context=False,
        parent_key=None,
    )
    if not isinstance(normalized_value, dict):
        raise AssertionError("Normalized baselineがobjectではありません")
    normalized = normalized_value
    json_path = directory / "baseline.json"
    markdown_path = directory / "baseline.md"
    write_atomic_json(json_path, normalized)
    digest = hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _write_atomic_text(
        markdown_path,
        _render_baseline_markdown(normalized, digest),
    )
    return json_path, markdown_path


def _normalize_baseline_value(
    value: object,
    *,
    within_execution_context: bool,
    parent_key: str | None,
) -> object:
    """baselineからattempt固有のsource revisionだけを除いて複製する。"""
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError("Acceptance baselineのkeyが文字列ではありません")
            child_within_execution_context = (
                within_execution_context or key == "execution_context"
            )
            if child_within_execution_context and key == "source_revision":
                continue
            if (
                within_execution_context
                and parent_key == "identity"
                and key == "commit"
            ):
                continue
            normalized[key] = _normalize_baseline_value(
                child,
                within_execution_context=child_within_execution_context,
                parent_key=key,
            )
        return normalized
    if isinstance(value, list):
        return [
            _normalize_baseline_value(
                item,
                within_execution_context=within_execution_context,
                parent_key=parent_key,
            )
            for item in value
        ]
    return value


def _write_atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    try:
        with temporary.open("w", encoding="utf-8") as file:
            file.write(value)
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _render_baseline_markdown(record: Mapping[str, object], digest: str) -> str:
    suite = record.get("suite")
    phases = record.get("phases")
    gates = record.get("automatic_gates")
    human = record.get("human_quality")
    return (
        "# Target acceptance baseline\n\n"
        f"- Schema: `{_BASELINE_SCHEMA}`\n"
        f"- Suite: `{suite}`\n"
        f"- Normalized digest: `{digest}`\n"
        f"- Phases: `{json.dumps(phases, sort_keys=True)}`\n"
        f"- Automatic gates: `{json.dumps(gates, sort_keys=True)}`\n"
        f"- Human quality: `{json.dumps(human, sort_keys=True)}`\n"
    )


def _validate_private_value(
    value: object,
    forbidden_values: Sequence[str],
    location: str,
) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or any(
                part in key.casefold() for part in _DENIED_KEY_PARTS
            ):
                msg = f"Acceptance recordにprivate keyがあります: {location}"
                raise ValueError(msg)
            _validate_private_value(item, forbidden_values, f"{location}.{key}")
        return
    if isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _validate_private_value(item, forbidden_values, f"{location}[{index}]")
        return
    if isinstance(value, str) and (
        string_looks_private(value)
        or _WINDOWS_PATH.search(value) is not None
        or any(forbidden and forbidden in value for forbidden in forbidden_values)
    ):
        raise ValueError(f"Acceptance recordにprivate valueがあります: {location}")


def _number(value: Mapping[str, object], key: str) -> float:
    result = value.get(key)
    if (
        not isinstance(result, int | float)
        or isinstance(result, bool)
        or not math.isfinite(result)
        or result < 0
    ):
        raise ValueError(
            f"Acceptance phase metric {key}が非負の有限numberではありません"
        )
    return float(result)


def _integer(value: Mapping[str, object], key: str) -> int:
    result = value.get(key)
    if not isinstance(result, int) or isinstance(result, bool) or result < 0:
        raise ValueError(f"Acceptance phase metric {key}が非負integerではありません")
    return result


def _boolean(value: Mapping[str, object], key: str) -> bool:
    result = value.get(key)
    if not isinstance(result, bool):
        raise ValueError(f"Acceptance phase metric {key}がbooleanではありません")
    return result


def _string(value: Mapping[str, object], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise ValueError(f"Acceptance phase metric {key}がstringではありません")
    return result


def _optional_string(value: Mapping[str, object], key: str) -> str | None:
    result = value.get(key)
    if result is None:
        return None
    if not isinstance(result, str) or not result:
        raise ValueError(
            f"Acceptance phase metric {key}がstringまたはnullではありません"
        )
    return result


def _comparison_gate(value: Mapping[str, object], key: str) -> bool:
    result = value.get(key)
    if not isinstance(result, bool):
        raise ValueError(f"Video Scan parallelism comparison gate {key}が不正です")
    return result
