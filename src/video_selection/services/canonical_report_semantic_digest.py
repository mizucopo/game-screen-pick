"""Canonical reportからrun固有診断を除いた意味digestを構築する。"""

import hashlib
import json
from collections.abc import Mapping
from typing import cast

_VOLATILE_RUN_KEYS = frozenset({"id", "started_at", "completed_at"})
_VOLATILE_RUN_WARNING_CODES = frozenset({"model_update_unavailable"})
_VOLATILE_RUNTIME_DIAGNOSTIC_KEYS = frozenset({"video_scan_parallelism"})
_VOLATILE_STAGE_DIAGNOSTIC_KEYS = frozenset(
    {
        "attempt_count",
        "cache_hits",
        "cache_misses",
        "duration_ms",
        "eval_tokens",
        "prompt_eval_tokens",
        "recomputed_items",
        "validation_failures",
    }
)
_VOLATILE_MODEL_LIFECYCLE_KEYS = frozenset(
    {
        "local_identity_before_update",
        "update_status",
    }
)


def canonical_report_semantic_digest(report: Mapping[str, object]) -> str:
    """再開前後で不変であるべきCanonical reportの意味digestを返す。"""
    canonical = json.dumps(
        _normalized_semantic_report(report),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _normalized_semantic_report(
    report: Mapping[str, object],
) -> dict[str, object]:
    """run固有identityと性能診断だけを除いたreport全体を返す。"""
    run = _mapping(report.get("run"), "Canonical report run")
    provenance = _mapping(
        report.get("provenance"),
        "Canonical report provenance",
    )
    stages = provenance.get("stages")
    if not isinstance(stages, list):
        raise ValueError("Canonical report provenance stagesが不正です")
    stage_mappings = [
        _mapping(value, "Canonical report provenance stage") for value in stages
    ]
    used_model_roles = {
        role
        for stage in stage_mappings
        for role in _string_list(
            stage.get("model_refs"),
            "Canonical report provenance stage model_refs",
        )
    }
    models = _mapping(
        provenance.get("models"),
        "Canonical report provenance models",
    )
    runtime = _mapping(
        provenance.get("runtime", {}),
        "Canonical report provenance runtime",
    )
    warnings = run.get("warnings")
    if not isinstance(warnings, list):
        raise ValueError("Canonical report run warningsが配列ではありません")
    semantic_warnings = [
        warning
        for warning in warnings
        if _mapping(
            warning,
            "Canonical report run warning",
        ).get("code")
        not in _VOLATILE_RUN_WARNING_CODES
    ]
    normalized = dict(report)
    normalized["run"] = {
        **{key: value for key, value in run.items() if key not in _VOLATILE_RUN_KEYS},
        "status": ("completed_with_warnings" if semantic_warnings else "completed"),
        "warnings": semantic_warnings,
    }
    normalized["provenance"] = {
        **provenance,
        "runtime": {
            key: value
            for key, value in runtime.items()
            if key not in _VOLATILE_RUNTIME_DIAGNOSTIC_KEYS
        },
        "models": {
            role: {
                key: item
                for key, item in _mapping(
                    model,
                    "Canonical report provenance model",
                ).items()
                if key not in _VOLATILE_MODEL_LIFECYCLE_KEYS
            }
            for role, model in models.items()
            if role in used_model_roles
        },
        "stages": [
            {
                key: item
                for key, item in _mapping(
                    value,
                    "Canonical report provenance stage",
                ).items()
                if key not in _VOLATILE_STAGE_DIAGNOSTIC_KEYS
            }
            for value in stage_mappings
        ],
    }
    return normalized


def _mapping(value: object, location: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{location}がobjectではありません")
    return cast(dict[str, object], value)


def _string_list(value: object, location: str) -> list[str]:
    """文字列配列を検証して返す。"""
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{location}が文字列配列ではありません")
    return value
